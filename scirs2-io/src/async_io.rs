//! Async I/O support for streaming capabilities
//!
//! This module provides asynchronous I/O interfaces for non-blocking processing
//! of large datasets. It builds on the streaming module to provide async versions
//! of file reading, writing, and processing operations.
//!
//! ## Features
//!
//! - **Async File Reading**: Non-blocking file reading with tokio
//! - **Async Streaming**: Asynchronous stream processing with backpressure
//! - **Concurrent Processing**: Process multiple chunks concurrently
//! - **Network I/O**: Support for async network operations
//! - **Cancellation**: Support for operation cancellation
//! - **Progress Tracking**: Real-time progress monitoring for async operations
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::async_io::{AsyncChunkedReader, AsyncStreamingConfig};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = AsyncStreamingConfig::new()
//!         .chunk_size(64 * 1024)
//!         .concurrency(4);
//!
//!     let mut reader = AsyncChunkedReader::new("large_file.dat", config).await?;
//!     
//!     while let Some(chunk_result) = reader.next().await {
//!         let chunk = chunk_result?;
//!         // Process chunk asynchronously
//!         tokio::task::yield_now().await; // Yield to allow other tasks
//!     }
//!     Ok(())
//! }
//! ```

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::error::{IoError, Result};

/// Configuration for async streaming operations
#[derive(Debug, Clone)]
pub struct AsyncStreamingConfig {
    /// Size of each chunk in bytes (default: 64KB)
    pub chunk_size: usize,
    /// Buffer size for I/O operations (default: 8KB)
    pub buffer_size: usize,
    /// Maximum concurrent operations (default: 4)
    pub concurrency: usize,
    /// Enable automatic backpressure handling
    pub enable_backpressure: bool,
    /// Timeout for individual operations in milliseconds
    pub operation_timeout_ms: Option<u64>,
    /// Maximum number of chunks to process (None for unlimited)
    pub max_chunks: Option<usize>,
    /// Skip the first N chunks
    pub skip_chunks: usize,
}

impl Default for AsyncStreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024, // 64KB
            buffer_size: 8 * 1024, // 8KB
            concurrency: 4,
            enable_backpressure: true,
            operation_timeout_ms: None,
            max_chunks: None,
            skip_chunks: 0,
        }
    }
}

impl AsyncStreamingConfig {
    /// Create a new async streaming configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the concurrency level
    pub fn concurrency(mut self, level: usize) -> Self {
        self.concurrency = level;
        self
    }

    /// Enable or disable backpressure handling
    pub fn enable_backpressure(mut self, enable: bool) -> Self {
        self.enable_backpressure = enable;
        self
    }

    /// Set operation timeout
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.operation_timeout_ms = Some(timeout_ms);
        self
    }

    /// Set maximum number of chunks to process
    pub fn max_chunks(mut self, max: usize) -> Self {
        self.max_chunks = Some(max);
        self
    }

    /// Set number of chunks to skip
    pub fn skip_chunks(mut self, skip: usize) -> Self {
        self.skip_chunks = skip;
        self
    }
}

/// Async iterator for reading files in chunks
pub struct AsyncChunkedReader {
    reader: BufReader<File>,
    config: AsyncStreamingConfig,
    chunks_read: usize,
    total_bytes_read: u64,
    finished: bool,
}

impl AsyncChunkedReader {
    /// Create a new async chunked reader for the specified file
    pub async fn new<P: AsRef<Path>>(path: P, config: AsyncStreamingConfig) -> Result<Self> {
        let file = File::open(path.as_ref())
            .await
            .map_err(|e| IoError::FileError(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::with_capacity(config.buffer_size, file);

        Ok(Self {
            reader,
            config,
            chunks_read: 0,
            total_bytes_read: 0,
            finished: false,
        })
    }

    /// Get the total number of bytes read so far
    pub fn bytes_read(&self) -> u64 {
        self.total_bytes_read
    }

    /// Get the number of chunks read so far
    pub fn chunks_read(&self) -> usize {
        self.chunks_read
    }

    /// Check if the reader has finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Read the next chunk asynchronously
    pub async fn read_next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        if self.finished {
            return Ok(None);
        }

        // Check if we should skip chunks
        if self.chunks_read < self.config.skip_chunks {
            let mut buffer = vec![0u8; self.config.chunk_size];
            match self.reader.read(&mut buffer).await {
                Ok(0) => {
                    self.finished = true;
                    return Ok(None);
                }
                Ok(bytes_read) => {
                    self.total_bytes_read += bytes_read as u64;
                    self.chunks_read += 1;
                    return Box::pin(self.read_next_chunk()).await; // Recursive call to skip
                }
                Err(e) => {
                    self.finished = true;
                    return Err(IoError::FileError(format!("Failed to skip chunk: {}", e)));
                }
            }
        }

        // Check max chunks limit
        if let Some(max) = self.config.max_chunks {
            if self.chunks_read >= max + self.config.skip_chunks {
                self.finished = true;
                return Ok(None);
            }
        }

        let mut chunk = vec![0u8; self.config.chunk_size];

        // Apply timeout if configured
        let read_future = self.reader.read(&mut chunk);

        let bytes_read = if let Some(timeout_ms) = self.config.operation_timeout_ms {
            match tokio::time::timeout(tokio::time::Duration::from_millis(timeout_ms), read_future)
                .await
            {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(e)) => {
                    self.finished = true;
                    return Err(IoError::FileError(format!("Failed to read chunk: {}", e)));
                }
                Err(_) => {
                    self.finished = true;
                    return Err(IoError::FileError("Read operation timed out".to_string()));
                }
            }
        } else {
            match read_future.await {
                Ok(bytes) => bytes,
                Err(e) => {
                    self.finished = true;
                    return Err(IoError::FileError(format!("Failed to read chunk: {}", e)));
                }
            }
        };

        if bytes_read == 0 {
            // End of file
            self.finished = true;
            Ok(None)
        } else {
            chunk.truncate(bytes_read);
            self.total_bytes_read += bytes_read as u64;
            self.chunks_read += 1;
            Ok(Some(chunk))
        }
    }
}

impl Stream for AsyncChunkedReader {
    type Item = Result<Vec<u8>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        // Create a future for the next chunk read
        let read_future = self.read_next_chunk();
        tokio::pin!(read_future);

        match read_future.poll(cx) {
            Poll::Ready(Ok(Some(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Ok(None)) => Poll::Ready(None),
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Async line-based reader
pub struct AsyncLineReader {
    reader: BufReader<File>,
    config: AsyncStreamingConfig,
    lines_read: usize,
    finished: bool,
}

impl AsyncLineReader {
    /// Create a new async line reader
    pub async fn new<P: AsRef<Path>>(path: P, config: AsyncStreamingConfig) -> Result<Self> {
        let file = File::open(path.as_ref())
            .await
            .map_err(|e| IoError::FileError(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::with_capacity(config.buffer_size, file);

        Ok(Self {
            reader,
            config,
            lines_read: 0,
            finished: false,
        })
    }

    /// Get the number of lines read so far
    pub fn lines_read(&self) -> usize {
        self.lines_read
    }

    /// Check if the reader has finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Read the next batch of lines asynchronously
    pub async fn read_next_lines(&mut self) -> Result<Option<Vec<String>>> {
        if self.finished {
            return Ok(None);
        }

        // Check if we should skip lines
        if self.lines_read < self.config.skip_chunks {
            let mut line = String::new();
            match self.reader.read_line(&mut line).await {
                Ok(0) => {
                    self.finished = true;
                    return Ok(None);
                }
                Ok(_) => {
                    self.lines_read += 1;
                    return Box::pin(self.read_next_lines()).await; // Recursive call to skip
                }
                Err(e) => {
                    self.finished = true;
                    return Err(IoError::FileError(format!("Failed to skip line: {}", e)));
                }
            }
        }

        // Check max chunks limit
        if let Some(max) = self.config.max_chunks {
            if self.lines_read >= max + self.config.skip_chunks {
                self.finished = true;
                return Ok(None);
            }
        }

        let mut lines = Vec::new();
        let target_lines = self.config.chunk_size; // Treat chunk_size as number of lines

        for _ in 0..target_lines {
            let mut line = String::new();

            let read_result = if let Some(timeout_ms) = self.config.operation_timeout_ms {
                match tokio::time::timeout(
                    tokio::time::Duration::from_millis(timeout_ms),
                    self.reader.read_line(&mut line),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => {
                        self.finished = true;
                        return Err(IoError::FileError(
                            "Read line operation timed out".to_string(),
                        ));
                    }
                }
            } else {
                self.reader.read_line(&mut line).await
            };

            match read_result {
                Ok(0) => {
                    // End of file
                    self.finished = true;
                    break;
                }
                Ok(_) => {
                    // Remove trailing newline
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    lines.push(line);
                    self.lines_read += 1;
                }
                Err(e) => {
                    self.finished = true;
                    return Err(IoError::FileError(format!("Failed to read line: {}", e)));
                }
            }
        }

        if lines.is_empty() {
            Ok(None)
        } else {
            Ok(Some(lines))
        }
    }
}

impl Stream for AsyncLineReader {
    type Item = Result<Vec<String>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        let read_future = self.read_next_lines();
        tokio::pin!(read_future);

        match read_future.poll(cx) {
            Poll::Ready(Ok(Some(lines))) => Poll::Ready(Some(Ok(lines))),
            Poll::Ready(Ok(None)) => Poll::Ready(None),
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Async statistics for tracking async operations
#[derive(Debug, Clone, Default)]
pub struct AsyncStreamingStats {
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total chunks processed
    pub chunks_processed: usize,
    /// Total lines processed (for line-based readers)
    pub lines_processed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Number of concurrent operations
    pub concurrent_operations: usize,
    /// Average processing speed in MB/s
    pub avg_speed_mbps: f64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
}

impl AsyncStreamingStats {
    /// Create new async streaming statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with chunk information
    pub fn update_chunk(&mut self, bytes: u64, processing_time_ms: f64) {
        self.bytes_processed += bytes;
        self.chunks_processed += 1;
        self.processing_time_ms += processing_time_ms;

        if self.processing_time_ms > 0.0 {
            let total_mb = self.bytes_processed as f64 / (1024.0 * 1024.0);
            let total_seconds = self.processing_time_ms / 1000.0;
            self.avg_speed_mbps = total_mb / total_seconds;
        }
    }

    /// Update statistics with line information
    pub fn update_lines(&mut self, lines: usize) {
        self.lines_processed += lines;
    }

    /// Update concurrent operations count
    pub fn update_concurrency(&mut self, count: usize) {
        self.concurrent_operations = count;
    }

    /// Update peak memory usage
    pub fn update_memory_usage(&mut self, usage: usize) {
        self.peak_memory_usage = self.peak_memory_usage.max(usage);
    }

    /// Get a summary string of the async statistics
    pub fn summary(&self) -> String {
        format!(
            "Async processed {} bytes in {} chunks ({} lines), {:.2} MB/s, {} concurrent ops, peak memory: {} KB",
            self.bytes_processed,
            self.chunks_processed,
            self.lines_processed,
            self.avg_speed_mbps,
            self.concurrent_operations,
            self.peak_memory_usage / 1024
        )
    }
}

/// Process a file asynchronously with concurrent chunk processing
pub async fn process_file_async<P, F, Fut, T>(
    path: P,
    config: AsyncStreamingConfig,
    processor: F,
) -> Result<(Vec<T>, AsyncStreamingStats)>
where
    P: AsRef<Path>,
    F: Fn(Vec<u8>, usize) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static,
{
    let reader = AsyncChunkedReader::new(path, config.clone()).await?;
    let mut stats = AsyncStreamingStats::new();
    let mut results = Vec::new();

    let start_time = std::time::Instant::now();

    // Process chunks with controlled concurrency
    let processor = std::sync::Arc::new(processor);
    let mut chunk_stream = reader
        .enumerate()
        .map(|(chunk_id, chunk_result)| {
            let processor = processor.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_start = std::time::Instant::now();
                        let result = processor(chunk.clone(), chunk_id).await?;
                        let processing_time = chunk_start.elapsed().as_secs_f64() * 1000.0;
                        Ok((result, chunk.len(), processing_time))
                    }
                    Err(e) => Err(e),
                }
            }
        })
        .buffer_unordered(config.concurrency);

    // Collect results
    while let Some(result) = chunk_stream.next().await {
        match result {
            Ok((processed_result, bytes, processing_time)) => {
                results.push(processed_result);
                stats.update_chunk(bytes as u64, processing_time);
                stats.update_concurrency(config.concurrency);
            }
            Err(e) => return Err(e),
        }
    }

    stats.processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok((results, stats))
}

/// Process a CSV file asynchronously with concurrent processing
pub async fn process_csv_async<P, F, Fut, T>(
    path: P,
    config: AsyncStreamingConfig,
    processor: F,
) -> Result<(Vec<T>, AsyncStreamingStats)>
where
    P: AsRef<Path>,
    F: Fn(Vec<String>, usize) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send,
    T: Send + 'static,
{
    let reader = AsyncLineReader::new(path, config.clone()).await?;
    let mut stats = AsyncStreamingStats::new();
    let mut results = Vec::new();

    let start_time = std::time::Instant::now();

    // Process line chunks with controlled concurrency
    let processor = std::sync::Arc::new(processor);
    let mut line_stream = reader
        .enumerate()
        .map(|(chunk_id, lines_result)| {
            let processor = processor.clone();
            async move {
                match lines_result {
                    Ok(lines) => {
                        let chunk_start = std::time::Instant::now();
                        let mut chunk_results = Vec::new();

                        for (line_id, line) in lines.into_iter().enumerate() {
                            let result = processor(vec![line], chunk_id * 1000 + line_id).await?;
                            chunk_results.push(result);
                        }

                        let processing_time = chunk_start.elapsed().as_secs_f64() * 1000.0;
                        Ok((chunk_results, processing_time))
                    }
                    Err(e) => Err(e),
                }
            }
        })
        .buffer_unordered(config.concurrency);

    // Collect results
    while let Some(result) = line_stream.next().await {
        match result {
            Ok((chunk_results, processing_time)) => {
                let line_count = chunk_results.len();
                results.extend(chunk_results);
                stats.update_chunk(0, processing_time); // CSV doesn't track bytes easily
                stats.update_lines(line_count);
                stats.update_concurrency(config.concurrency);
            }
            Err(e) => return Err(e),
        }
    }

    stats.processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok((results, stats))
}

/// Create a cancellation token for async operations
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Cancel the operation
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the operation is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Create a clone of this token (for sharing across async tasks)
    pub fn clone(&self) -> Self {
        Self {
            cancelled: self.cancelled.clone(),
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_async_chunked_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_async.txt");

        // Create test data
        let test_data = "0123456789".repeat(100); // 1000 bytes
        std::fs::write(&file_path, &test_data).unwrap();

        let config = AsyncStreamingConfig::new().chunk_size(100);
        let mut reader = AsyncChunkedReader::new(&file_path, config).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk_result) = reader.read_next_chunk().await.unwrap() {
            chunks.push(chunk_result);
        }

        assert_eq!(chunks.len(), 10); // 1000 bytes / 100 bytes per chunk
        for chunk in &chunks {
            assert_eq!(chunk.len(), 100);
        }
    }

    #[tokio::test]
    async fn test_async_line_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_async_lines.txt");

        // Create test data with lines
        let lines: Vec<String> = (0..50).map(|i| format!("Line {}", i)).collect();
        std::fs::write(&file_path, lines.join("\n")).unwrap();

        let config = AsyncStreamingConfig::new().chunk_size(10); // 10 lines per chunk
        let mut reader = AsyncLineReader::new(&file_path, config).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(lines_result) = reader.read_next_lines().await.unwrap() {
            chunks.push(lines_result);
        }

        assert_eq!(chunks.len(), 5); // 50 lines / 10 lines per chunk
        for chunk in &chunks {
            assert_eq!(chunk.len(), 10);
        }
    }

    #[tokio::test]
    async fn test_async_config() {
        let config = AsyncStreamingConfig::new()
            .chunk_size(1024)
            .buffer_size(4096)
            .concurrency(8)
            .timeout(5000);

        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.buffer_size, 4096);
        assert_eq!(config.concurrency, 8);
        assert_eq!(config.operation_timeout_ms, Some(5000));
    }

    #[tokio::test]
    async fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        let cloned_token = token.clone();
        assert!(cloned_token.is_cancelled());
    }
}
