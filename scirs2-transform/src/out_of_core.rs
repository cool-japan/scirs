//! Out-of-core processing for large datasets
//!
//! This module provides utilities for processing datasets that are too large
//! to fit in memory, using chunked processing and memory-mapped files.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{Result, TransformError};
use crate::normalize::NormalizationMethod;

/// Configuration for out-of-core processing
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum chunk size in MB
    pub chunk_size_mb: usize,
    /// Whether to use memory mapping when possible
    pub use_mmap: bool,
    /// Number of threads for parallel processing
    pub n_threads: usize,
    /// Temporary directory for intermediate files
    pub temp_dir: String,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        OutOfCoreConfig {
            chunk_size_mb: 100,
            use_mmap: true,
            n_threads: num_cpus::get(),
            temp_dir: std::env::temp_dir().to_string_lossy().to_string(),
        }
    }
}

/// Trait for transformers that support out-of-core processing
pub trait OutOfCoreTransformer: Send + Sync {
    /// Fit the transformer on chunks of data
    fn fit_chunks<I>(&mut self, chunks: I) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>;

    /// Transform data in chunks
    fn transform_chunks<I>(&self, chunks: I) -> Result<ChunkedArrayWriter>
    where
        I: Iterator<Item = Result<Array2<f64>>>;

    /// Get the expected shape of transformed data
    fn get_transformshape(&self, inputshape: (usize, usize)) -> (usize, usize);
}

/// Reader for chunked array data from disk
pub struct ChunkedArrayReader {
    file: BufReader<File>,
    shape: (usize, usize),
    chunk_size: usize,
    current_row: usize,
    dtype_size: usize,
    /// Reusable buffer for reading data to reduce allocations
    buffer_pool: Vec<u8>,
}

impl ChunkedArrayReader {
    /// Create a new chunked array reader
    pub fn new<P: AsRef<Path>>(path: P, shape: (usize, usize), chunk_size: usize) -> Result<Self> {
        let file = File::open(&path).map_err(|e| {
            TransformError::TransformationError(format!("Failed to open file: {e}"))
        })?;

        // Pre-allocate buffer pool for maximum chunk _size
        let max_chunk_bytes = chunk_size * shape.1 * std::mem::size_of::<f64>();

        Ok(ChunkedArrayReader {
            file: BufReader::new(file),
            shape,
            chunk_size,
            current_row: 0,
            dtype_size: std::mem::size_of::<f64>(),
            buffer_pool: vec![0u8; max_chunk_bytes],
        })
    }

    /// Read the next chunk of data with optimized bulk reading and buffer reuse
    pub fn read_chunk(&mut self) -> Result<Option<Array2<f64>>> {
        if self.current_row >= self.shape.0 {
            return Ok(None);
        }

        let rows_to_read = (self.chunk_size).min(self.shape.0 - self.current_row);
        let mut chunk = Array2::zeros((rows_to_read, self.shape.1));

        // Use pre-allocated buffer to avoid allocation overhead
        let total_elements = rows_to_read * self.shape.1;
        let total_bytes = total_elements * self.dtype_size;

        // Ensure buffer is large enough (it should be from constructor)
        if self.buffer_pool.len() < total_bytes {
            return Err(TransformError::TransformationError(
                "Buffer pool too small for chunk".to_string(),
            ));
        }

        // Read into pre-allocated buffer
        self.file
            .read_exact(&mut self.buffer_pool[..total_bytes])
            .map_err(|e| {
                TransformError::TransformationError(format!("Failed to read data: {e}"))
            })?;

        // Convert bytes to f64 values efficiently using chunks iterator
        for (element_idx, f64_bytes) in self.buffer_pool[..total_bytes].chunks_exact(8).enumerate()
        {
            let i = element_idx / self.shape.1;
            let j = element_idx % self.shape.1;

            // Convert 8 bytes to f64 safely
            let mut bytes_array = [0u8; 8];
            bytes_array.copy_from_slice(f64_bytes);
            chunk[[i, j]] = f64::from_le_bytes(bytes_array);
        }

        self.current_row += rows_to_read;
        Ok(Some(chunk))
    }

    /// Create an iterator over chunks
    pub fn chunks(self) -> ChunkedArrayIterator {
        ChunkedArrayIterator { reader: self }
    }
}

/// Iterator over chunks of array data
pub struct ChunkedArrayIterator {
    reader: ChunkedArrayReader,
}

impl Iterator for ChunkedArrayIterator {
    type Item = Result<Array2<f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Writer for chunked array data to disk
#[derive(Debug)]
pub struct ChunkedArrayWriter {
    file: BufWriter<File>,
    shape: (usize, usize),
    rows_written: usize,
    path: String,
    /// Reusable buffer for writing data to reduce allocations
    write_buffer: Vec<u8>,
}

impl ChunkedArrayWriter {
    /// Create a new chunked array writer
    pub fn new<P: AsRef<Path>>(path: P, shape: (usize, usize)) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::create(&path).map_err(|e| {
            TransformError::TransformationError(format!("Failed to create file: {e}"))
        })?;

        // Pre-allocate write buffer for typical chunk sizes (e.g., 1000 rows)
        let typical_chunk_size = 1000_usize.min(shape.0);
        let buffer_capacity = typical_chunk_size * shape.1 * std::mem::size_of::<f64>();

        Ok(ChunkedArrayWriter {
            file: BufWriter::new(file),
            shape,
            rows_written: 0,
            path: path_str,
            write_buffer: Vec::with_capacity(buffer_capacity),
        })
    }

    /// Write a chunk of data with optimized bulk writing
    pub fn write_chunk(&mut self, chunk: &Array2<f64>) -> Result<()> {
        if chunk.shape()[1] != self.shape.1 {
            return Err(TransformError::InvalidInput(format!(
                "Chunk has {} columns, expected {}",
                chunk.shape()[1],
                self.shape.1
            )));
        }

        if self.rows_written + chunk.shape()[0] > self.shape.0 {
            return Err(TransformError::InvalidInput(
                "Too many rows written".to_string(),
            ));
        }

        // Optimize bulk writing by batching bytes
        let total_elements = chunk.shape()[0] * chunk.shape()[1];
        let total_bytes = total_elements * std::mem::size_of::<f64>();

        // Ensure buffer has enough capacity
        self.write_buffer.clear();
        self.write_buffer.reserve(total_bytes);

        // Convert all f64 values to bytes in batch
        for i in 0..chunk.shape()[0] {
            for j in 0..chunk.shape()[1] {
                let bytes = chunk[[i, j]].to_le_bytes();
                self.write_buffer.extend_from_slice(&bytes);
            }
        }

        // Write all bytes at once
        self.file.write_all(&self.write_buffer).map_err(|e| {
            TransformError::TransformationError(format!("Failed to write data: {e}"))
        })?;

        self.rows_written += chunk.shape()[0];
        Ok(())
    }

    /// Finalize the writer and flush data
    pub fn finalize(mut self) -> Result<String> {
        self.file.flush().map_err(|e| {
            TransformError::TransformationError(format!("Failed to flush data: {e}"))
        })?;

        if self.rows_written != self.shape.0 {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} rows, but wrote {}",
                self.shape.0, self.rows_written
            )));
        }

        Ok(self.path)
    }
}

/// Out-of-core normalizer implementation
pub struct OutOfCoreNormalizer {
    method: NormalizationMethod,
    // Statistics computed during fit
    stats: Option<NormalizationStats>,
}

#[derive(Clone)]
struct NormalizationStats {
    min: Array1<f64>,
    max: Array1<f64>,
    mean: Array1<f64>,
    std: Array1<f64>,
    median: Array1<f64>,
    iqr: Array1<f64>,
    count: usize,
}

impl OutOfCoreNormalizer {
    /// Create a new out-of-core normalizer
    pub fn new(method: NormalizationMethod) -> Self {
        OutOfCoreNormalizer {
            method,
            stats: None,
        }
    }

    /// Compute statistics in a single pass for simple methods
    fn compute_simple_stats<I>(&mut self, chunks: I, nfeatures: usize) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        let mut min = Array1::from_elem(nfeatures, f64::INFINITY);
        let mut max = Array1::from_elem(nfeatures, f64::NEG_INFINITY);
        let mut sum = Array1::zeros(nfeatures);
        let mut sum_sq = Array1::zeros(nfeatures);
        let mut count = 0;

        // First pass: compute min, max, sum, sum_sq
        for chunk_result in chunks {
            let chunk = chunk_result?;
            count += chunk.shape()[0];

            for j in 0..nfeatures {
                let col = chunk.column(j);
                for &val in col.iter() {
                    min[j] = min[j].min(val);
                    max[j] = max[j].max(val);
                    sum[j] += val;
                    sum_sq[j] += val * val;
                }
            }
        }

        // Compute mean and std
        let mean = sum / count as f64;
        let variance = sum_sq / count as f64 - &mean * &mean;
        let std = variance.mapv(|v: f64| v.sqrt());

        self.stats = Some(NormalizationStats {
            min,
            max,
            mean,
            std,
            median: Array1::zeros(nfeatures), // Not used for simple methods
            iqr: Array1::zeros(nfeatures),    // Not used for simple methods
            count,
        });

        Ok(())
    }

    /// Compute robust statistics using approximate quantile estimation
    fn compute_robust_stats<I>(&mut self, chunks: I, nfeatures: usize) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        // Use reservoir sampling to approximate quantiles
        const RESERVOIR_SIZE: usize = 10000; // Sample size for quantile estimation

        let mut reservoirs: Vec<Vec<f64>> = vec![Vec::with_capacity(RESERVOIR_SIZE); nfeatures];
        let mut count = 0;
        let mut rng = scirs2_core::random::rng();

        // First pass: build reservoirs using reservoir sampling
        for chunk_result in chunks {
            let chunk = chunk_result?;

            for i in 0..chunk.shape()[0] {
                count += 1;

                for j in 0..nfeatures {
                    let val = chunk[[i, j]];

                    if reservoirs[j].len() < RESERVOIR_SIZE {
                        // Reservoir not full, just add the element
                        reservoirs[j].push(val);
                    } else {
                        // Reservoir full, randomly replace with decreasing probability
                        let k = (count as f64 * rng.random::<f64>()) as usize;
                        if k < RESERVOIR_SIZE {
                            reservoirs[j][k] = val;
                        }
                    }
                }
            }
        }

        // Compute median and IQR from reservoirs
        let mut median = Array1::zeros(nfeatures);
        let mut iqr = Array1::zeros(nfeatures);

        for j in 0..nfeatures {
            if !reservoirs[j].is_empty() {
                reservoirs[j].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let len = reservoirs[j].len();

                // Median (50th percentile)
                median[j] = if len.is_multiple_of(2) {
                    (reservoirs[j][len / 2 - 1] + reservoirs[j][len / 2]) / 2.0
                } else {
                    reservoirs[j][len / 2]
                };

                // First quartile (25th percentile)
                let q1_idx = len / 4;
                let q1 = reservoirs[j][q1_idx];

                // Third quartile (75th percentile)
                let q3_idx = 3 * len / 4;
                let q3 = reservoirs[j][q3_idx.min(len - 1)];

                // Interquartile range
                iqr[j] = q3 - q1;

                // Ensure IQR is not zero (add small epsilon for numerical stability)
                if iqr[j] < 1e-10 {
                    iqr[j] = 1.0;
                }
            } else {
                median[j] = 0.0;
                iqr[j] = 1.0;
            }
        }

        self.stats = Some(NormalizationStats {
            min: Array1::zeros(nfeatures),  // Not used for robust scaling
            max: Array1::zeros(nfeatures),  // Not used for robust scaling
            mean: Array1::zeros(nfeatures), // Not used for robust scaling
            std: Array1::zeros(nfeatures),  // Not used for robust scaling
            median,
            iqr,
            count,
        });

        Ok(())
    }
}

impl OutOfCoreTransformer for OutOfCoreNormalizer {
    fn fit_chunks<I>(&mut self, chunks: I) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        // Peek at the first chunk to get dimensions
        let mut chunks_iter = chunks.peekable();
        let nfeatures = match chunks_iter.peek() {
            Some(Ok(chunk)) => chunk.shape()[1],
            Some(Err(_)) => return chunks_iter.next().unwrap().map(|_| ()),
            None => {
                return Err(TransformError::InvalidInput(
                    "No chunks provided".to_string(),
                ))
            }
        };

        match self.method {
            NormalizationMethod::MinMax
            | NormalizationMethod::MinMaxCustom(_, _)
            | NormalizationMethod::ZScore
            | NormalizationMethod::MaxAbs => {
                self.compute_simple_stats(chunks_iter, nfeatures)?;
            }
            NormalizationMethod::Robust => {
                // Robust scaling using approximate quantile estimation
                self.compute_robust_stats(chunks_iter, nfeatures)?;
            }
            _ => {
                return Err(TransformError::NotImplemented(
                    "This normalization method is not supported for out-of-core processing"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    fn transform_chunks<I>(&self, chunks: I) -> Result<ChunkedArrayWriter>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        if self.stats.is_none() {
            return Err(TransformError::TransformationError(
                "Normalizer has not been fitted".to_string(),
            ));
        }

        let stats = self.stats.as_ref().unwrap();

        // Create temporary output file
        let output_path = format!(
            "{}/transform_output_{}.bin",
            std::env::temp_dir().to_string_lossy(),
            std::process::id()
        );

        let mut writer = ChunkedArrayWriter::new(&output_path, (stats.count, stats.min.len()))?;

        // Transform each chunk
        for chunk_result in chunks {
            let chunk = chunk_result?;
            let mut transformed = Array2::zeros((chunk.nrows(), chunk.ncols()));

            match self.method {
                NormalizationMethod::MinMax => {
                    let range = &stats.max - &stats.min;
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            if range[j].abs() > 1e-10 {
                                transformed[[i, j]] = (chunk[[i, j]] - stats.min[j]) / range[j];
                            } else {
                                transformed[[i, j]] = 0.5;
                            }
                        }
                    }
                }
                NormalizationMethod::ZScore => {
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            if stats.std[j] > 1e-10 {
                                transformed[[i, j]] =
                                    (chunk[[i, j]] - stats.mean[j]) / stats.std[j];
                            } else {
                                transformed[[i, j]] = 0.0;
                            }
                        }
                    }
                }
                NormalizationMethod::MaxAbs => {
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            let max_abs = stats.max[j].abs().max(stats.min[j].abs());
                            if max_abs > 1e-10 {
                                transformed[[i, j]] = chunk[[i, j]] / max_abs;
                            } else {
                                transformed[[i, j]] = 0.0;
                            }
                        }
                    }
                }
                NormalizationMethod::Robust => {
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            // Robust scaling: (x - median) / IQR
                            if stats.iqr[j] > 1e-10 {
                                transformed[[i, j]] =
                                    (chunk[[i, j]] - stats.median[j]) / stats.iqr[j];
                            } else {
                                transformed[[i, j]] = 0.0;
                            }
                        }
                    }
                }
                _ => {
                    return Err(TransformError::NotImplemented(
                        "This normalization method is not supported".to_string(),
                    ));
                }
            }

            writer.write_chunk(&transformed)?;
        }

        Ok(writer)
    }

    fn get_transformshape(&self, inputshape: (usize, usize)) -> (usize, usize) {
        inputshape // Normalization doesn't change shape
    }
}

/// Create chunks from a large CSV file
#[allow(dead_code)]
pub fn csv_chunks<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
    has_header: bool,
) -> Result<impl Iterator<Item = Result<Array2<f64>>>> {
    let file = File::open(path).map_err(|e| {
        TransformError::TransformationError(format!("Failed to open CSV file: {e}"))
    })?;

    Ok(CsvChunkIterator::new(
        BufReader::new(file),
        chunk_size,
        has_header,
    ))
}

/// Iterator that reads CSV in chunks
struct CsvChunkIterator {
    reader: BufReader<File>,
    chunk_size: usize,
    skipheader: bool,
    header_skipped: bool,
}

impl CsvChunkIterator {
    fn new(_reader: BufReader<File>, chunk_size: usize, skipheader: bool) -> Self {
        CsvChunkIterator {
            reader: _reader,
            chunk_size,
            skipheader,
            header_skipped: false,
        }
    }
}

impl Iterator for CsvChunkIterator {
    type Item = Result<Array2<f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::BufRead;

        let mut rows = Vec::new();
        let mut n_cols = None;

        for line_result in (&mut self.reader).lines().take(self.chunk_size) {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => return Some(Err(TransformError::IoError(e))),
            };

            // Skip header if needed
            if self.skipheader && !self.header_skipped {
                self.header_skipped = true;
                continue;
            }

            // Parse CSV line
            let values: Result<Vec<f64>> = line
                .split(',')
                .map(|s| {
                    s.trim().parse::<f64>().map_err(|e| {
                        TransformError::ParseError(format!("Failed to parse number: {e}"))
                    })
                })
                .collect();

            let values = match values {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };

            // Check column consistency
            if let Some(nc) = n_cols {
                if values.len() != nc {
                    return Some(Err(TransformError::InvalidInput(
                        "Inconsistent number of columns in CSV".to_string(),
                    )));
                }
            } else {
                n_cols = Some(values.len());
            }

            rows.push(values);
        }

        if rows.is_empty() {
            return None;
        }

        // Convert to Array2
        let n_rows = rows.len();
        let n_cols = n_cols.unwrap();
        let mut array = Array2::zeros((n_rows, n_cols));

        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                array[[i, j]] = val;
            }
        }

        Some(Ok(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_out_of_core_robust_scaling() {
        // Create test data with known quantiles
        let data = vec![
            Array::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap(),
            Array::from_shape_vec((3, 2), vec![4.0, 40.0, 5.0, 50.0, 6.0, 60.0]).unwrap(),
            Array::from_shape_vec((3, 2), vec![7.0, 70.0, 8.0, 80.0, 9.0, 90.0]).unwrap(),
        ];

        // Create chunks iterator
        let chunks = data.into_iter().map(|chunk| Ok(chunk));

        // Fit robust normalizer
        let mut normalizer = OutOfCoreNormalizer::new(NormalizationMethod::Robust);
        normalizer.fit_chunks(chunks).unwrap();

        // Check that statistics were computed
        let stats = normalizer.stats.as_ref().unwrap();
        assert_eq!(stats.median.len(), 2);
        assert_eq!(stats.iqr.len(), 2);

        // For the first column: [1,2,3,4,5,6,7,8,9]
        // Median should be around 5.0, IQR should be around 4.0 (Q3=7.5, Q1=2.5)
        assert!((stats.median[0] - 5.0).abs() < 1.0); // Allow some approximation error
        assert!(stats.iqr[0] > 0.0);

        // For the second column: [10,20,30,40,50,60,70,80,90]
        // Median should be around 50.0, IQR should be around 40.0
        assert!((stats.median[1] - 50.0).abs() < 10.0); // Allow some approximation error
        assert!(stats.iqr[1] > 0.0);
    }

    #[test]
    fn test_out_of_core_robust_transform() {
        // Create simple test data
        let fit_data = vec![
            Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap(),
            Array::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap(),
            Array::from_shape_vec((1, 1), vec![5.0]).unwrap(),
        ];

        let mut normalizer = OutOfCoreNormalizer::new(NormalizationMethod::Robust);
        normalizer
            .fit_chunks(fit_data.into_iter().map(|chunk| Ok(chunk)))
            .unwrap();

        // Transform new data
        let transform_data = vec![Array::from_shape_vec((2, 1), vec![3.0, 6.0]).unwrap()];

        let result = normalizer.transform_chunks(transform_data.into_iter().map(|chunk| Ok(chunk)));
        assert!(result.is_ok());
    }

    #[test]
    fn test_out_of_core_normalizer_not_fitted() {
        let normalizer = OutOfCoreNormalizer::new(NormalizationMethod::Robust);
        let data = vec![Array::zeros((2, 2))];

        let result = normalizer.transform_chunks(data.into_iter().map(|chunk| Ok(chunk)));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not been fitted"));
    }

    #[test]
    fn test_out_of_core_empty_chunks() {
        let mut normalizer = OutOfCoreNormalizer::new(NormalizationMethod::Robust);
        let empty_chunks: Vec<Result<Array2<f64>>> = vec![];

        let result = normalizer.fit_chunks(empty_chunks.into_iter());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chunks provided"));
    }
}
