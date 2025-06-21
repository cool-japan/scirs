//! Streaming and iterator interfaces for large data handling
//!
//! This module provides memory-efficient streaming interfaces for processing large datasets
//! without loading everything into memory at once. It includes iterator-based APIs for
//! reading data in chunks, streaming decompression, and incremental processing.
//!
//! ## Features
//!
//! - **Chunked Reading**: Read large files in configurable chunks
//! - **Streaming Decompression**: Decompress data on-the-fly while reading
//! - **Iterator Interfaces**: Process data using Rust's iterator paradigm
//! - **Memory Efficiency**: Minimize memory usage for large dataset processing
//! - **Parallel Processing**: Combine with rayon for parallel chunk processing
//! - **Format Support**: Streaming support for CSV, Matrix Market, and binary formats
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::streaming::{ChunkedReader, StreamingConfig};
//! use std::path::Path;
//!
//! // Read a large CSV file in 1MB chunks
//! let config = StreamingConfig::new()
//!     .chunk_size(1024 * 1024)  // 1MB chunks
//!     .buffer_size(8192);       // 8KB buffer
//!
//! let reader = ChunkedReader::new("large_data.csv", config)?;
//!
//! for (chunk_id, chunk_data) in reader.enumerate() {
//!     let data = chunk_data?;
//!     println!("Processing chunk {}: {} bytes", chunk_id, data.len());
//!     // Process data without loading entire file
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::compression::CompressionAlgorithm;
use crate::error::{IoError, Result};

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each chunk in bytes (default: 64KB)
    pub chunk_size: usize,
    /// Buffer size for I/O operations (default: 8KB)
    pub buffer_size: usize,
    /// Enable automatic compression detection
    pub auto_detect_compression: bool,
    /// Compression algorithm (if known)
    pub compression: Option<CompressionAlgorithm>,
    /// Maximum number of chunks to process (None for unlimited)
    pub max_chunks: Option<usize>,
    /// Skip the first N chunks
    pub skip_chunks: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024, // 64KB
            buffer_size: 8 * 1024, // 8KB
            auto_detect_compression: true,
            compression: None,
            max_chunks: None,
            skip_chunks: 0,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration with default values
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

    /// Enable or disable automatic compression detection
    pub fn auto_detect_compression(mut self, enable: bool) -> Self {
        self.auto_detect_compression = enable;
        self
    }

    /// Set the compression algorithm
    pub fn compression(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.compression = Some(algorithm);
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

/// Iterator for reading files in chunks
pub struct ChunkedReader {
    reader: BufReader<File>,
    config: StreamingConfig,
    chunks_read: usize,
    total_bytes_read: u64,
    finished: bool,
}

impl ChunkedReader {
    /// Create a new chunked reader for the specified file
    pub fn new<P: AsRef<Path>>(path: P, config: StreamingConfig) -> Result<Self> {
        let file = File::open(path.as_ref())
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

    /// Skip the specified number of bytes
    pub fn skip_bytes(&mut self, bytes: u64) -> Result<u64> {
        let skipped = self
            .reader
            .seek(SeekFrom::Current(bytes as i64))
            .map_err(|e| IoError::FileError(format!("Failed to skip bytes: {}", e)))?;
        self.total_bytes_read += bytes;
        Ok(skipped)
    }

    /// Get the file position
    pub fn position(&mut self) -> Result<u64> {
        self.reader
            .stream_position()
            .map_err(|e| IoError::FileError(format!("Failed to get position: {}", e)))
    }
}

impl Iterator for ChunkedReader {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Check if we should skip chunks
        if self.chunks_read < self.config.skip_chunks {
            match self.skip_bytes(self.config.chunk_size as u64) {
                Ok(_) => {
                    self.chunks_read += 1;
                    return self.next(); // Recursive call to skip
                }
                Err(e) => return Some(Err(e)),
            }
        }

        // Check max chunks limit
        if let Some(max) = self.config.max_chunks {
            if self.chunks_read >= max + self.config.skip_chunks {
                self.finished = true;
                return None;
            }
        }

        let mut chunk = vec![0u8; self.config.chunk_size];
        match self.reader.read(&mut chunk) {
            Ok(0) => {
                // End of file
                self.finished = true;
                None
            }
            Ok(bytes_read) => {
                chunk.truncate(bytes_read);
                self.total_bytes_read += bytes_read as u64;
                self.chunks_read += 1;
                Some(Ok(chunk))
            }
            Err(e) => {
                self.finished = true;
                Some(Err(IoError::FileError(format!(
                    "Failed to read chunk: {}",
                    e
                ))))
            }
        }
    }
}

/// Iterator for reading lines from a file in chunks
pub struct LineChunkedReader {
    reader: BufReader<File>,
    config: StreamingConfig,
    lines_read: usize,
    finished: bool,
}

impl LineChunkedReader {
    /// Create a new line-based chunked reader
    pub fn new<P: AsRef<Path>>(path: P, config: StreamingConfig) -> Result<Self> {
        let file = File::open(path.as_ref())
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
}

impl Iterator for LineChunkedReader {
    type Item = Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Check if we should skip lines
        if self.lines_read < self.config.skip_chunks {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    return None;
                }
                Ok(_) => {
                    self.lines_read += 1;
                    return self.next(); // Recursive call to skip
                }
                Err(e) => {
                    return Some(Err(IoError::FileError(format!(
                        "Failed to skip line: {}",
                        e
                    ))))
                }
            }
        }

        // Check max chunks limit
        if let Some(max) = self.config.max_chunks {
            if self.lines_read >= max + self.config.skip_chunks {
                self.finished = true;
                return None;
            }
        }

        let mut lines = Vec::new();
        let target_lines = self.config.chunk_size; // Treat chunk_size as number of lines

        for _ in 0..target_lines {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
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
                    return Some(Err(IoError::FileError(format!(
                        "Failed to read line: {}",
                        e
                    ))));
                }
            }
        }

        if lines.is_empty() {
            None
        } else {
            Some(Ok(lines))
        }
    }
}

/// Streaming CSV reader that processes rows in chunks
pub struct StreamingCsvReader {
    line_reader: LineChunkedReader,
    header: Option<Vec<String>>,
    delimiter: char,
    has_header: bool,
}

impl StreamingCsvReader {
    /// Create a new streaming CSV reader
    pub fn new<P: AsRef<Path>>(path: P, config: StreamingConfig) -> Result<Self> {
        let line_reader = LineChunkedReader::new(path, config)?;

        Ok(Self {
            line_reader,
            header: None,
            delimiter: ',',
            has_header: false,
        })
    }

    /// Set the delimiter character
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Enable header row processing
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Get the header row (if available)
    pub fn header(&self) -> Option<&Vec<String>> {
        self.header.as_ref()
    }

    /// Parse a CSV line into fields
    fn parse_line(&self, line: &str) -> Vec<String> {
        // Simple CSV parsing - in production, you'd use a proper CSV parser
        line.split(self.delimiter)
            .map(|field| field.trim().to_string())
            .collect()
    }
}

impl Iterator for StreamingCsvReader {
    type Item = Result<Vec<Vec<String>>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Handle header if not yet processed
        if self.has_header && self.header.is_none() {
            match self.line_reader.next() {
                Some(Ok(lines)) => {
                    if let Some(header_line) = lines.first() {
                        self.header = Some(self.parse_line(header_line));
                    }
                    // Continue to process remaining lines in this chunk
                    let data_lines: Vec<Vec<String>> = lines
                        .iter()
                        .skip(1)
                        .map(|line| self.parse_line(line))
                        .collect();

                    if data_lines.is_empty() {
                        return self.next(); // Get next chunk
                    } else {
                        return Some(Ok(data_lines));
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            }
        }

        // Process regular data chunks
        match self.line_reader.next() {
            Some(Ok(lines)) => {
                let data_rows: Vec<Vec<String>> =
                    lines.iter().map(|line| self.parse_line(line)).collect();
                Some(Ok(data_rows))
            }
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }
}

/// Statistics for streaming operations
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total chunks processed
    pub chunks_processed: usize,
    /// Total lines processed (for line-based readers)
    pub lines_processed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Average bytes per chunk
    pub avg_bytes_per_chunk: f64,
    /// Average processing speed in MB/s
    pub avg_speed_mbps: f64,
}

impl StreamingStats {
    /// Create new streaming statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with chunk information
    pub fn update_chunk(&mut self, bytes: u64, processing_time_ms: f64) {
        self.bytes_processed += bytes;
        self.chunks_processed += 1;
        self.processing_time_ms += processing_time_ms;

        self.avg_bytes_per_chunk = self.bytes_processed as f64 / self.chunks_processed as f64;

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

    /// Get a summary string of the statistics
    pub fn summary(&self) -> String {
        format!(
            "Processed {} bytes in {} chunks ({} lines), avg {:.2} MB/s",
            self.bytes_processed, self.chunks_processed, self.lines_processed, self.avg_speed_mbps
        )
    }
}

/// Process a file using a streaming reader with progress tracking
pub fn process_file_chunked<P, F, T>(
    path: P,
    config: StreamingConfig,
    mut processor: F,
) -> Result<(T, StreamingStats)>
where
    P: AsRef<Path>,
    F: FnMut(&[u8], usize) -> Result<T>,
    T: Default,
{
    let reader = ChunkedReader::new(path, config)?;
    let mut stats = StreamingStats::new();
    let mut result = T::default();

    let start_time = std::time::Instant::now();

    for (chunk_id, chunk_result) in reader.enumerate() {
        let chunk_start = std::time::Instant::now();

        match chunk_result {
            Ok(chunk_data) => {
                result = processor(&chunk_data, chunk_id)?;

                let chunk_time = chunk_start.elapsed().as_secs_f64() * 1000.0;
                stats.update_chunk(chunk_data.len() as u64, chunk_time);
            }
            Err(e) => return Err(e),
        }
    }

    stats.processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok((result, stats))
}

/// Process a CSV file using streaming with progress tracking
pub fn process_csv_chunked<P, F, T>(
    path: P,
    config: StreamingConfig,
    has_header: bool,
    mut processor: F,
) -> Result<(T, StreamingStats)>
where
    P: AsRef<Path>,
    F: FnMut(&[Vec<String>], usize, Option<&Vec<String>>) -> Result<T>,
    T: Default,
{
    let mut reader = StreamingCsvReader::new(path, config)?.with_header(has_header);
    let mut stats = StreamingStats::new();
    let mut result = T::default();

    let start_time = std::time::Instant::now();

    let mut chunk_id = 0;
    while let Some(chunk_result) = reader.next() {
        let chunk_start = std::time::Instant::now();

        match chunk_result {
            Ok(rows) => {
                let header = reader.header();
                result = processor(&rows, chunk_id, header)?;

                let chunk_time = chunk_start.elapsed().as_secs_f64() * 1000.0;
                stats.update_chunk(0, chunk_time); // CSV doesn't track bytes easily
                stats.update_lines(rows.len());
                chunk_id += 1;
            }
            Err(e) => return Err(e),
        }
    }

    stats.processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok((result, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_chunked_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_data.txt");

        // Create test data
        let test_data = "0123456789".repeat(100); // 1000 bytes
        std::fs::write(&file_path, &test_data).unwrap();

        let config = StreamingConfig::new().chunk_size(100);
        let reader = ChunkedReader::new(&file_path, config).unwrap();

        let chunks: Result<Vec<_>> = reader.collect();
        let chunks = chunks.unwrap();

        assert_eq!(chunks.len(), 10); // 1000 bytes / 100 bytes per chunk
        for chunk in &chunks {
            assert_eq!(chunk.len(), 100);
        }
    }

    #[test]
    fn test_line_chunked_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_lines.txt");

        // Create test data with lines
        let lines: Vec<String> = (0..50).map(|i| format!("Line {}", i)).collect();
        std::fs::write(&file_path, lines.join("\n")).unwrap();

        let config = StreamingConfig::new().chunk_size(10); // 10 lines per chunk
        let reader = LineChunkedReader::new(&file_path, config).unwrap();

        let chunks: Result<Vec<_>> = reader.collect();
        let chunks = chunks.unwrap();

        assert_eq!(chunks.len(), 5); // 50 lines / 10 lines per chunk
        for chunk in &chunks {
            assert_eq!(chunk.len(), 10);
        }
    }

    #[test]
    fn test_streaming_csv_reader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.csv");

        // Create test CSV
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "name,age,city").unwrap();
        for i in 0..20 {
            writeln!(file, "Person{},{},City{}", i, 20 + i, i % 5).unwrap();
        }

        let config = StreamingConfig::new().chunk_size(5); // 5 lines per chunk
        let reader = StreamingCsvReader::new(&file_path, config)
            .unwrap()
            .with_header(true);

        let chunks: Result<Vec<_>> = reader.collect();
        let chunks = chunks.unwrap();

        // With 1 header + 20 data rows, chunk size 5:
        // First chunk: header + 4 data rows -> returns 4 data rows
        // Remaining chunks: 5, 5, 5, 1 data rows -> 4 more chunks
        // Total: 5 chunks (4 + 5 + 5 + 5 + 1 = 20 data rows)
        assert_eq!(chunks.len(), 5);

        // Verify total data rows
        let total_rows: usize = chunks.iter().map(|chunk| chunk.len()).sum();
        assert_eq!(total_rows, 20);

        // Verify all rows have 3 columns
        for chunk in &chunks {
            for row in chunk {
                assert_eq!(row.len(), 3); // 3 columns
            }
        }
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::new()
            .chunk_size(1024)
            .buffer_size(4096)
            .max_chunks(10)
            .skip_chunks(2);

        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.buffer_size, 4096);
        assert_eq!(config.max_chunks, Some(10));
        assert_eq!(config.skip_chunks, 2);
    }

    #[test]
    fn test_process_file_chunked() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_process.txt");

        // Create test data
        let test_data = "Hello World!".repeat(100);
        std::fs::write(&file_path, &test_data).unwrap();

        let config = StreamingConfig::new().chunk_size(100);

        let (total_size, stats) =
            process_file_chunked(&file_path, config, |chunk, _chunk_id| -> Result<usize> {
                Ok(chunk.len())
            })
            .unwrap();

        assert_eq!(total_size, 100); // Last chunk size
        assert!(stats.bytes_processed > 0);
        assert!(stats.chunks_processed > 0);
    }
}
