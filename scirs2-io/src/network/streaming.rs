//! Streaming I/O operations for efficient network data transfer
//!
//! This module provides streaming capabilities for handling large files over network
//! connections with minimal memory usage. It supports chunked reading/writing,
//! progress monitoring, and efficient buffering strategies.

use crate::error::{IoError, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for streaming operations
    pub buffer_size: usize,
    /// Maximum memory usage for buffering
    pub max_memory: usize,
    /// Enable compression during streaming
    pub compression: bool,
    /// Progress reporting interval (in bytes)
    pub progress_interval: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,       // 64KB chunks
            max_memory: 16 * 1024 * 1024, // 16MB max buffer
            compression: false,
            progress_interval: 1024 * 1024, // Report every 1MB
        }
    }
}

/// Progress information for streaming operations
#[derive(Debug, Clone)]
pub struct StreamProgress {
    /// Bytes transferred so far
    pub bytes_transferred: u64,
    /// Total bytes to transfer (if known)
    pub total_bytes: Option<u64>,
    /// Transfer rate in bytes per second
    pub rate: f64,
    /// Estimated time remaining (if total known)
    pub eta_seconds: Option<f64>,
}

impl StreamProgress {
    /// Calculate progress percentage (0-100)
    pub fn percentage(&self) -> Option<f64> {
        self.total_bytes.map(|total| {
            if total > 0 {
                (self.bytes_transferred as f64 / total as f64) * 100.0
            } else {
                0.0
            }
        })
    }

    /// Check if transfer is complete
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_bytes {
            self.bytes_transferred >= total
        } else {
            false
        }
    }
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(StreamProgress) + Send + Sync>;

/// Stream reader with progress tracking
pub struct ProgressReader<R: Read> {
    inner: R,
    bytes_read: u64,
    total_bytes: Option<u64>,
    progress_callback: Option<ProgressCallback>,
    progress_interval: u64,
    last_progress_report: u64,
    start_time: std::time::Instant,
}

impl<R: Read> ProgressReader<R> {
    /// Create a new progress reader
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            bytes_read: 0,
            total_bytes: None,
            progress_callback: None,
            progress_interval: 1024 * 1024, // 1MB
            last_progress_report: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Set total bytes for progress calculation
    pub fn with_total_bytes(mut self, total: u64) -> Self {
        self.total_bytes = Some(total);
        self
    }

    /// Set progress callback
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Set progress reporting interval
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Get current progress
    pub fn progress(&self) -> StreamProgress {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            self.bytes_read as f64 / elapsed
        } else {
            0.0
        };

        let eta_seconds = if let Some(total) = self.total_bytes {
            if rate > 0.0 && self.bytes_read < total {
                Some((total - self.bytes_read) as f64 / rate)
            } else {
                None
            }
        } else {
            None
        };

        StreamProgress {
            bytes_transferred: self.bytes_read,
            total_bytes: self.total_bytes,
            rate,
            eta_seconds,
        }
    }

    fn report_progress(&mut self) {
        if let Some(ref callback) = self.progress_callback {
            let progress = self.progress();
            callback(progress);
            self.last_progress_report = self.bytes_read;
        }
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        self.bytes_read += bytes_read as u64;

        // Report progress if interval reached
        if self.bytes_read - self.last_progress_report >= self.progress_interval {
            self.report_progress();
        }

        Ok(bytes_read)
    }
}

/// Stream writer with progress tracking
pub struct ProgressWriter<W: Write> {
    inner: W,
    bytes_written: u64,
    total_bytes: Option<u64>,
    progress_callback: Option<ProgressCallback>,
    progress_interval: u64,
    last_progress_report: u64,
    start_time: std::time::Instant,
}

impl<W: Write> ProgressWriter<W> {
    /// Create a new progress writer
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
            total_bytes: None,
            progress_callback: None,
            progress_interval: 1024 * 1024, // 1MB
            last_progress_report: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Set total bytes for progress calculation
    pub fn with_total_bytes(mut self, total: u64) -> Self {
        self.total_bytes = Some(total);
        self
    }

    /// Set progress callback
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Set progress reporting interval
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Get current progress
    pub fn progress(&self) -> StreamProgress {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            self.bytes_written as f64 / elapsed
        } else {
            0.0
        };

        let eta_seconds = if let Some(total) = self.total_bytes {
            if rate > 0.0 && self.bytes_written < total {
                Some((total - self.bytes_written) as f64 / rate)
            } else {
                None
            }
        } else {
            None
        };

        StreamProgress {
            bytes_transferred: self.bytes_written,
            total_bytes: self.total_bytes,
            rate,
            eta_seconds,
        }
    }

    fn report_progress(&mut self) {
        if let Some(ref callback) = self.progress_callback {
            let progress = self.progress();
            callback(progress);
            self.last_progress_report = self.bytes_written;
        }
    }
}

impl<W: Write> Write for ProgressWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let bytes_written = self.inner.write(buf)?;
        self.bytes_written += bytes_written as u64;

        // Report progress if interval reached
        if self.bytes_written - self.last_progress_report >= self.progress_interval {
            self.report_progress();
        }

        Ok(bytes_written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Chunked file reader for streaming large files
pub struct ChunkedReader {
    file: std::fs::File,
    chunk_size: usize,
    current_position: u64,
    file_size: u64,
}

impl ChunkedReader {
    /// Create a new chunked reader
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to open file: {}", e)))?;

        let file_size = file
            .metadata()
            .map_err(|e| IoError::FileError(format!("Failed to get file metadata: {}", e)))?
            .len();

        Ok(Self {
            file,
            chunk_size,
            current_position: 0,
            file_size,
        })
    }

    /// Read the next chunk
    pub fn read_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        if self.current_position >= self.file_size {
            return Ok(None);
        }

        let mut buffer = vec![0u8; self.chunk_size];
        let bytes_read = self
            .file
            .read(&mut buffer)
            .map_err(|e| IoError::FileError(format!("Failed to read chunk: {}", e)))?;

        if bytes_read == 0 {
            return Ok(None);
        }

        buffer.truncate(bytes_read);
        self.current_position += bytes_read as u64;

        Ok(Some(buffer))
    }

    /// Seek to a specific position
    pub fn seek(&mut self, position: u64) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(position))
            .map_err(|e| IoError::FileError(format!("Failed to seek: {}", e)))?;
        self.current_position = position;
        Ok(())
    }

    /// Get current position
    pub fn position(&self) -> u64 {
        self.current_position
    }

    /// Get file size
    pub fn size(&self) -> u64 {
        self.file_size
    }

    /// Check if at end of file
    pub fn is_eof(&self) -> bool {
        self.current_position >= self.file_size
    }

    /// Get progress percentage
    pub fn progress_percentage(&self) -> f64 {
        if self.file_size > 0 {
            (self.current_position as f64 / self.file_size as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Chunked file writer for streaming large files
pub struct ChunkedWriter {
    file: std::fs::File,
    bytes_written: u64,
    buffer: Vec<u8>,
    buffer_size: usize,
}

impl ChunkedWriter {
    /// Create a new chunked writer
    pub fn new<P: AsRef<Path>>(path: P, buffer_size: usize) -> Result<Self> {
        let file = std::fs::File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;

        Ok(Self {
            file,
            bytes_written: 0,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        })
    }

    /// Write a chunk of data
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(data);

        // Flush buffer if it's full
        if self.buffer.len() >= self.buffer_size {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Flush the internal buffer
    pub fn flush_buffer(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            self.file
                .write_all(&self.buffer)
                .map_err(|e| IoError::FileError(format!("Failed to write buffer: {}", e)))?;

            self.bytes_written += self.buffer.len() as u64;
            self.buffer.clear();
        }
        Ok(())
    }

    /// Finish writing and close the file
    pub fn finish(mut self) -> Result<u64> {
        self.flush_buffer()?;
        self.file
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush file: {}", e)))?;
        Ok(self.bytes_written)
    }

    /// Get bytes written so far
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written + self.buffer.len() as u64
    }
}

/// Stream copy with progress tracking
pub fn copy_with_progress<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    total_size: Option<u64>,
    progress_callback: Option<ProgressCallback>,
) -> Result<u64> {
    let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer
    let mut total_copied = 0u64;
    let start_time = std::time::Instant::now();
    let mut last_progress_report = 0u64;
    let progress_interval = 1024 * 1024; // Report every 1MB

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| IoError::FileError(format!("Read error: {}", e)))?;

        if bytes_read == 0 {
            break;
        }

        writer
            .write_all(&buffer[..bytes_read])
            .map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;

        total_copied += bytes_read as u64;

        // Report progress if needed
        if let Some(ref callback) = progress_callback {
            if total_copied - last_progress_report >= progress_interval {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    total_copied as f64 / elapsed
                } else {
                    0.0
                };

                let eta_seconds = if let Some(total) = total_size {
                    if rate > 0.0 && total_copied < total {
                        Some((total - total_copied) as f64 / rate)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let progress = StreamProgress {
                    bytes_transferred: total_copied,
                    total_bytes: total_size,
                    rate,
                    eta_seconds,
                };

                callback(progress);
                last_progress_report = total_copied;
            }
        }
    }

    // Final progress report
    if let Some(ref callback) = progress_callback {
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            total_copied as f64 / elapsed
        } else {
            0.0
        };

        let progress = StreamProgress {
            bytes_transferred: total_copied,
            total_bytes: total_size,
            rate,
            eta_seconds: Some(0.0),
        };

        callback(progress);
    }

    Ok(total_copied)
}

/// Async stream copy with progress tracking
#[cfg(feature = "async")]
pub async fn async_copy_with_progress<R, W>(
    mut reader: R,
    mut writer: W,
    total_size: Option<u64>,
    progress_callback: Option<ProgressCallback>,
) -> Result<u64>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer
    let mut total_copied = 0u64;
    let start_time = std::time::Instant::now();
    let mut last_progress_report = 0u64;
    let progress_interval = 1024 * 1024; // Report every 1MB

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .await
            .map_err(|e| IoError::FileError(format!("Async read error: {}", e)))?;

        if bytes_read == 0 {
            break;
        }

        writer
            .write_all(&buffer[..bytes_read])
            .await
            .map_err(|e| IoError::FileError(format!("Async write error: {}", e)))?;

        total_copied += bytes_read as u64;

        // Report progress if needed
        if let Some(ref callback) = progress_callback {
            if total_copied - last_progress_report >= progress_interval {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    total_copied as f64 / elapsed
                } else {
                    0.0
                };

                let eta_seconds = if let Some(total) = total_size {
                    if rate > 0.0 && total_copied < total {
                        Some((total - total_copied) as f64 / rate)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let progress = StreamProgress {
                    bytes_transferred: total_copied,
                    total_bytes: total_size,
                    rate,
                    eta_seconds,
                };

                callback(progress);
                last_progress_report = total_copied;
            }
        }
    }

    writer
        .flush()
        .await
        .map_err(|e| IoError::FileError(format!("Async flush error: {}", e)))?;

    Ok(total_copied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, 64 * 1024);
        assert_eq!(config.max_memory, 16 * 1024 * 1024);
        assert!(!config.compression);
        assert_eq!(config.progress_interval, 1024 * 1024);
    }

    #[test]
    fn test_stream_progress() {
        let progress = StreamProgress {
            bytes_transferred: 512,
            total_bytes: Some(1024),
            rate: 256.0,
            eta_seconds: Some(2.0),
        };

        assert_eq!(progress.percentage(), Some(50.0));
        assert!(!progress.is_complete());

        let complete_progress = StreamProgress {
            bytes_transferred: 1024,
            total_bytes: Some(1024),
            rate: 512.0,
            eta_seconds: Some(0.0),
        };

        assert_eq!(complete_progress.percentage(), Some(100.0));
        assert!(complete_progress.is_complete());
    }

    #[test]
    fn test_progress_reader() {
        let data = b"Hello, world! This is test data for streaming.";
        let cursor = Cursor::new(data);

        let mut reader = ProgressReader::new(cursor)
            .with_total_bytes(data.len() as u64)
            .with_progress_interval(10);

        let mut buffer = [0u8; 20];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, 20);
        assert_eq!(reader.progress().bytes_transferred, 20);

        let progress = reader.progress();
        assert_eq!(progress.bytes_transferred, 20);
        assert_eq!(progress.total_bytes, Some(data.len() as u64));
        assert!(progress.rate >= 0.0);
    }

    #[test]
    fn test_progress_writer() {
        let mut output = Vec::new();
        let mut writer = ProgressWriter::new(&mut output)
            .with_total_bytes(100)
            .with_progress_interval(25);

        let data = b"Test data for progress writer functionality.";
        let bytes_written = writer.write(data).unwrap();

        assert_eq!(bytes_written, data.len());
        assert_eq!(writer.progress().bytes_transferred, data.len() as u64);

        writer.flush().unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_chunked_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_chunked.txt");

        // Create test file
        let test_data = b"This is test data for chunked reading. It should be read in chunks.";
        std::fs::write(&file_path, test_data).unwrap();

        let mut reader = ChunkedReader::new(&file_path, 10).unwrap();
        assert_eq!(reader.size(), test_data.len() as u64);
        assert!(!reader.is_eof());

        let mut all_data = Vec::new();
        while let Some(chunk) = reader.read_chunk().unwrap() {
            all_data.extend_from_slice(&chunk);
        }

        assert_eq!(all_data, test_data);
        assert!(reader.is_eof());
        assert_eq!(reader.progress_percentage(), 100.0);
    }

    #[test]
    fn test_chunked_writer() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_chunked_write.txt");

        let mut writer = ChunkedWriter::new(&file_path, 20).unwrap();

        let data1 = b"First chunk of data.";
        let data2 = b"Second chunk of data.";

        writer.write_chunk(data1).unwrap();
        writer.write_chunk(data2).unwrap();

        let total_bytes = writer.finish().unwrap();
        assert_eq!(total_bytes, (data1.len() + data2.len()) as u64);

        // Verify file contents
        let file_contents = std::fs::read(&file_path).unwrap();
        let expected = [&data1[..], &data2[..]].concat();
        assert_eq!(file_contents, expected);
    }

    #[test]
    fn test_copy_with_progress() {
        let input_data = b"This is test data for copy with progress functionality. It demonstrates streaming copy operations.";
        let input = Cursor::new(input_data);
        let mut output = Vec::new();

        let progress_reports = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let progress_reports_clone = progress_reports.clone();
        let callback = Box::new(move |progress: StreamProgress| {
            progress_reports_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(progress.bytes_transferred <= input_data.len() as u64);
            assert!(progress.rate >= 0.0);
        }) as ProgressCallback;

        // Use a small progress interval to ensure we get reports
        let copied = copy_with_progress(
            input,
            &mut output,
            Some(input_data.len() as u64),
            Some(callback),
        )
        .unwrap();

        assert_eq!(copied, input_data.len() as u64);
        assert_eq!(output, input_data);
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn test_async_copy_with_progress() {
        let input_data = b"Async test data for copy with progress functionality.";
        let input = Cursor::new(input_data);
        let mut output = Vec::new();

        let copied =
            async_copy_with_progress(input, &mut output, Some(input_data.len() as u64), None)
                .await
                .unwrap();

        assert_eq!(copied, input_data.len() as u64);
        assert_eq!(output, input_data);
    }
}
