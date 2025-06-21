//! # Compressed Memory Buffers
//!
//! This module provides compressed memory buffer implementations for memory-constrained environments.
//! It supports various compression algorithms optimized for scientific data patterns.

use crate::error::CoreError;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use lz4::{Decoder as Lz4Decoder, EncoderBuilder as Lz4EncoderBuilder};
use ndarray::{Array, ArrayBase, Data, Dimension};
use std::io::Result as IoResult;
use std::io::{Read, Write};
use std::marker::PhantomData;

/// Compression algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// Gzip compression - good balance of compression ratio and speed
    Gzip,
    /// LZ4 compression - faster compression/decompression with moderate compression ratio
    Lz4,
    /// No compression - passthrough for testing or when compression is not beneficial
    None,
}

/// Compression level settings
#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    /// Fastest compression with lowest compression ratio
    Fast,
    /// Balanced compression and speed
    Default,
    /// Best compression ratio but slower
    Best,
    /// Custom compression level (0-9 for gzip, 0-12 for LZ4)
    Custom(u32),
}

impl From<CompressionLevel> for u32 {
    fn from(level: CompressionLevel) -> Self {
        match level {
            CompressionLevel::Fast => 1,
            CompressionLevel::Default => 6,
            CompressionLevel::Best => 9,
            CompressionLevel::Custom(level) => level,
        }
    }
}

/// Compressed buffer for storing scientific data with automatic compression/decompression
pub struct CompressedBuffer<T> {
    compressed_data: Vec<u8>,
    algorithm: CompressionAlgorithm,
    #[allow(dead_code)]
    compression_level: CompressionLevel,
    original_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> CompressedBuffer<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new compressed buffer from raw data
    pub fn new(
        data: &[T],
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> IoResult<Self> {
        let original_size = std::mem::size_of_val(data);
        let bytes = bytemuck::cast_slice(data);

        let compressed_data = match algorithm {
            CompressionAlgorithm::Gzip => Self::compress_gzip(bytes, level)?,
            CompressionAlgorithm::Lz4 => Self::compress_lz4(bytes, level)?,
            CompressionAlgorithm::None => bytes.to_vec(),
        };

        Ok(Self {
            compressed_data,
            algorithm,
            compression_level: level,
            original_size,
            _phantom: PhantomData,
        })
    }

    /// Decompress and return the original data
    pub fn decompress(&self) -> IoResult<Vec<T>> {
        let decompressed_bytes = match self.algorithm {
            CompressionAlgorithm::Gzip => Self::decompress_gzip(&self.compressed_data)?,
            CompressionAlgorithm::Lz4 => Self::decompress_lz4(&self.compressed_data)?,
            CompressionAlgorithm::None => self.compressed_data.clone(),
        };

        // Verify the size matches expectations
        if decompressed_bytes.len() != self.original_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Decompressed data size doesn't match original size",
            ));
        }

        let data = bytemuck::cast_slice(&decompressed_bytes).to_vec();
        Ok(data)
    }

    /// Get the compression ratio (original_size / compressed_size)
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.compressed_data.len() as f64
    }

    /// Get the compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        self.compressed_data.len()
    }

    /// Get the original size in bytes
    pub fn original_size(&self) -> usize {
        self.original_size
    }

    /// Get the compression algorithm used
    pub fn algorithm(&self) -> CompressionAlgorithm {
        self.algorithm
    }

    fn compress_gzip(data: &[u8], level: CompressionLevel) -> IoResult<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level.into()));
        encoder.write_all(data)?;
        encoder.finish()
    }

    fn decompress_gzip(data: &[u8]) -> IoResult<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    fn compress_lz4(data: &[u8], level: CompressionLevel) -> IoResult<Vec<u8>> {
        let mut encoder = Lz4EncoderBuilder::new()
            .level(std::cmp::min(level.into(), 12) as u32)
            .build(Vec::new())?;
        encoder.write_all(data)?;
        Ok(encoder.finish().0)
    }

    fn decompress_lz4(data: &[u8]) -> IoResult<Vec<u8>> {
        let mut decoder = Lz4Decoder::new(data)?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}

/// Compressed array wrapper for ndarray types
pub struct CompressedArray<T, D>
where
    D: Dimension,
{
    buffer: CompressedBuffer<T>,
    shape: D,
}

impl<T, D> CompressedArray<T, D>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone,
    D: Dimension,
{
    /// Create a compressed array from an ndarray
    pub fn from_array<S>(
        array: &ArrayBase<S, D>,
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = T>,
    {
        let data = if array.is_standard_layout() {
            // Can use the underlying data directly
            array.as_slice().unwrap().to_vec()
        } else {
            // Need to collect into a contiguous layout
            array.iter().cloned().collect()
        };

        let buffer = CompressedBuffer::new(&data, algorithm, level).map_err(|e| {
            CoreError::CompressionError(crate::error::ErrorContext::new(e.to_string()))
        })?;

        Ok(Self {
            buffer,
            shape: array.raw_dim(),
        })
    }

    /// Decompress and reconstruct the original array
    pub fn to_array(&self) -> Result<Array<T, D>, CoreError> {
        let data = self.buffer.decompress().map_err(|e| {
            CoreError::CompressionError(crate::error::ErrorContext::new(e.to_string()))
        })?;

        Array::from_shape_vec(self.shape.clone(), data)
            .map_err(|e| CoreError::InvalidShape(crate::error::ErrorContext::new(e.to_string())))
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.buffer.compression_ratio()
    }

    /// Get the compressed size
    pub fn compressed_size(&self) -> usize {
        self.buffer.compressed_size()
    }

    /// Get the original size
    pub fn original_size(&self) -> usize {
        self.buffer.original_size()
    }

    /// Get the array shape
    pub const fn shape(&self) -> &D {
        &self.shape
    }
}

/// Compressed memory pool for managing multiple compressed buffers
pub struct CompressedBufferPool<T> {
    buffers: Vec<CompressedBuffer<T>>,
    algorithm: CompressionAlgorithm,
    compression_level: CompressionLevel,
    total_original_size: usize,
    total_compressed_size: usize,
}

impl<T> CompressedBufferPool<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new compressed buffer pool
    pub fn new(algorithm: CompressionAlgorithm, level: CompressionLevel) -> Self {
        Self {
            buffers: Vec::new(),
            algorithm,
            compression_level: level,
            total_original_size: 0,
            total_compressed_size: 0,
        }
    }

    /// Add a buffer to the pool
    pub fn add_buffer(&mut self, data: &[T]) -> IoResult<usize> {
        let buffer = CompressedBuffer::new(data, self.algorithm, self.compression_level)?;
        self.total_original_size += buffer.original_size();
        self.total_compressed_size += buffer.compressed_size();
        let buffer_id = self.buffers.len();
        self.buffers.push(buffer);
        Ok(buffer_id)
    }

    /// Get a buffer by ID
    pub fn get_buffer(&self, id: usize) -> Option<&CompressedBuffer<T>> {
        self.buffers.get(id)
    }

    /// Remove a buffer from the pool
    pub fn remove_buffer(&mut self, id: usize) -> Option<CompressedBuffer<T>> {
        if id < self.buffers.len() {
            let buffer = self.buffers.swap_remove(id);
            self.total_original_size -= buffer.original_size();
            self.total_compressed_size -= buffer.compressed_size();
            Some(buffer)
        } else {
            None
        }
    }

    /// Get the total compression ratio for all buffers
    pub fn total_compression_ratio(&self) -> f64 {
        if self.total_compressed_size == 0 {
            1.0
        } else {
            self.total_original_size as f64 / self.total_compressed_size as f64
        }
    }

    /// Get the total memory saved (original - compressed)
    pub fn memory_saved(&self) -> usize {
        self.total_original_size
            .saturating_sub(self.total_compressed_size)
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> CompressionStats {
        CompressionStats {
            buffer_count: self.buffers.len(),
            total_original_size: self.total_original_size,
            total_compressed_size: self.total_compressed_size,
            compression_ratio: self.total_compression_ratio(),
            memory_saved: self.memory_saved(),
            algorithm: self.algorithm,
        }
    }

    /// Clear all buffers from the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.total_original_size = 0;
        self.total_compressed_size = 0;
    }
}

/// Statistics about compression performance
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub buffer_count: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub compression_ratio: f64,
    pub memory_saved: usize,
    pub algorithm: CompressionAlgorithm,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compression Stats:\n\
             - Algorithm: {:?}\n\
             - Buffers: {}\n\
             - Original Size: {} bytes ({:.2} MB)\n\
             - Compressed Size: {} bytes ({:.2} MB)\n\
             - Compression Ratio: {:.2}x\n\
             - Memory Saved: {} bytes ({:.2} MB)",
            self.algorithm,
            self.buffer_count,
            self.total_original_size,
            self.total_original_size as f64 / 1024.0 / 1024.0,
            self.total_compressed_size,
            self.total_compressed_size as f64 / 1024.0 / 1024.0,
            self.compression_ratio,
            self.memory_saved,
            self.memory_saved as f64 / 1024.0 / 1024.0
        )
    }
}

/// Adaptive compression that chooses the best algorithm based on data characteristics
pub struct AdaptiveCompression;

impl AdaptiveCompression {
    /// Choose the best compression algorithm for the given data
    pub fn choose_algorithm<T>(data: &[T]) -> CompressionAlgorithm
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        let bytes = bytemuck::cast_slice(data);

        // Sample compression ratios with different algorithms
        let sample_size = std::cmp::min(bytes.len(), 4096); // Sample first 4KB
        let sample = &bytes[..sample_size];

        let gzip_ratio = Self::estimate_compression_ratio(sample, CompressionAlgorithm::Gzip);
        let lz4_ratio = Self::estimate_compression_ratio(sample, CompressionAlgorithm::Lz4);

        // Choose based on compression ratio threshold
        if gzip_ratio > 2.0 {
            CompressionAlgorithm::Gzip
        } else if lz4_ratio > 1.5 {
            CompressionAlgorithm::Lz4
        } else {
            CompressionAlgorithm::None
        }
    }

    fn estimate_compression_ratio(data: &[u8], algorithm: CompressionAlgorithm) -> f64 {
        match algorithm {
            CompressionAlgorithm::Gzip => {
                if let Ok(compressed) =
                    CompressedBuffer::<u8>::compress_gzip(data, CompressionLevel::Fast)
                {
                    data.len() as f64 / compressed.len() as f64
                } else {
                    1.0
                }
            }
            CompressionAlgorithm::Lz4 => {
                if let Ok(compressed) =
                    CompressedBuffer::<u8>::compress_lz4(data, CompressionLevel::Fast)
                {
                    data.len() as f64 / compressed.len() as f64
                } else {
                    1.0
                }
            }
            CompressionAlgorithm::None => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_compressed_buffer_basic() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        let buffer =
            CompressedBuffer::new(&data, CompressionAlgorithm::Gzip, CompressionLevel::Default)
                .expect("Failed to create compressed buffer");

        let decompressed = buffer.decompress().expect("Failed to decompress");
        assert_eq!(data, decompressed);
        assert!(buffer.compression_ratio() > 1.0);
    }

    #[test]
    fn test_compressed_array() {
        let array = Array2::<f64>::zeros((100, 100));

        let compressed =
            CompressedArray::from_array(&array, CompressionAlgorithm::Lz4, CompressionLevel::Fast)
                .expect("Failed to create compressed array");

        let decompressed = compressed.to_array().expect("Failed to decompress array");
        assert_eq!(array, decompressed);
    }

    #[test]
    fn test_compressed_buffer_pool() {
        let mut pool =
            CompressedBufferPool::new(CompressionAlgorithm::Gzip, CompressionLevel::Default);

        let data1: Vec<f32> = vec![1.0; 1000];
        let data2: Vec<f32> = (0..1000).map(|i| i as f32).collect();

        let id1 = pool.add_buffer(&data1).expect("Failed to add buffer 1");
        let _id2 = pool.add_buffer(&data2).expect("Failed to add buffer 2");

        assert_eq!(pool.stats().buffer_count, 2);
        assert!(pool.total_compression_ratio() > 1.0);

        let buffer1 = pool.get_buffer(id1).expect("Failed to get buffer 1");
        let decompressed1 = buffer1.decompress().expect("Failed to decompress buffer 1");
        assert_eq!(data1, decompressed1);
    }

    #[test]
    fn test_adaptive_compression() {
        // Test with highly compressible data (zeros)
        let compressible_data: Vec<f64> = vec![0.0; 10000];
        let algorithm = AdaptiveCompression::choose_algorithm(&compressible_data);
        assert!(matches!(algorithm, CompressionAlgorithm::Gzip));

        // Test with random data (less compressible)
        let random_data: Vec<u8> = (0..1000).map(|i| (i * 17 + 42) as u8).collect();
        let algorithm = AdaptiveCompression::choose_algorithm(&random_data);
        // This might be Lz4 or None depending on the specific data pattern
        assert!(matches!(
            algorithm,
            CompressionAlgorithm::Lz4 | CompressionAlgorithm::None
        ));
    }

    #[test]
    fn test_compression_levels() {
        let data: Vec<f64> = vec![1.0; 1000];

        // Test all compression levels
        let levels = vec![
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
            CompressionLevel::Custom(5),
        ];

        for level in levels {
            let buffer = CompressedBuffer::new(&data, CompressionAlgorithm::Gzip, level)
                .expect("Failed to create buffer");
            let decompressed = buffer.decompress().expect("Failed to decompress");
            assert_eq!(data, decompressed);
        }
    }

    #[test]
    fn test_compression_level_conversion() {
        assert_eq!(u32::from(CompressionLevel::Fast), 1);
        assert_eq!(u32::from(CompressionLevel::Default), 6);
        assert_eq!(u32::from(CompressionLevel::Best), 9);
        assert_eq!(u32::from(CompressionLevel::Custom(7)), 7);
    }

    #[test]
    fn test_all_compression_algorithms() {
        let data: Vec<u32> = (0..100).collect();

        let algorithms = vec![
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::None,
        ];

        for algo in algorithms {
            let buffer = CompressedBuffer::new(&data, algo, CompressionLevel::Default)
                .expect("Failed to create buffer");

            assert_eq!(buffer.algorithm(), algo);
            assert_eq!(buffer.original_size(), std::mem::size_of_val(&data));

            let decompressed = buffer.decompress().expect("Failed to decompress");
            assert_eq!(data, decompressed);

            if algo == CompressionAlgorithm::None {
                assert_eq!(buffer.compression_ratio(), 1.0);
            }
        }
    }

    #[test]
    fn test_compressed_buffer_lz4() {
        let data: Vec<i32> = (0..5000).map(|i| i * 2).collect();

        let buffer =
            CompressedBuffer::new(&data, CompressionAlgorithm::Lz4, CompressionLevel::Fast)
                .expect("Failed to create LZ4 buffer");

        let decompressed = buffer.decompress().expect("Failed to decompress");
        assert_eq!(data, decompressed);
        assert!(buffer.compressed_size() < buffer.original_size());
    }

    #[test]
    fn test_compressed_array_non_standard_layout() {
        // Create a transposed array (non-standard layout)
        let array = Array2::<f64>::from_shape_fn((50, 50), |(i, j)| (i * 50 + j) as f64);
        let transposed = array.t();

        let compressed = CompressedArray::from_array(
            &transposed,
            CompressionAlgorithm::Gzip,
            CompressionLevel::Default,
        )
        .expect("Failed to create compressed array");

        let decompressed = compressed.to_array().expect("Failed to decompress");
        assert_eq!(transposed, decompressed);
        assert_eq!(compressed.shape().slice(), transposed.shape());
    }

    #[test]
    fn test_compressed_buffer_pool_operations() {
        let mut pool = CompressedBufferPool::new(CompressionAlgorithm::Lz4, CompressionLevel::Fast);

        // Test empty pool
        assert_eq!(pool.stats().buffer_count, 0);
        assert_eq!(pool.total_compression_ratio(), 1.0);
        assert_eq!(pool.memory_saved(), 0);

        // Add buffers
        let data1: Vec<f64> = vec![0.0; 500];
        let data2: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let data3: Vec<f64> = vec![std::f64::consts::PI; 500];

        let id1 = pool.add_buffer(&data1).expect("Failed to add buffer 1");
        let id2 = pool.add_buffer(&data2).expect("Failed to add buffer 2");
        let id3 = pool.add_buffer(&data3).expect("Failed to add buffer 3");

        assert_eq!(pool.stats().buffer_count, 3);

        // Test get_buffer
        assert!(pool.get_buffer(id1).is_some());
        assert!(pool.get_buffer(id2).is_some());
        assert!(pool.get_buffer(id3).is_some());
        assert!(pool.get_buffer(100).is_none());

        // Test remove_buffer
        let removed = pool.remove_buffer(id2).expect("Failed to remove buffer");
        let decompressed = removed.decompress().expect("Failed to decompress");
        assert_eq!(data2, decompressed);
        assert_eq!(pool.stats().buffer_count, 2);

        // Test remove non-existent buffer
        assert!(pool.remove_buffer(100).is_none());

        // Test clear
        pool.clear();
        assert_eq!(pool.stats().buffer_count, 0);
        assert_eq!(pool.total_compression_ratio(), 1.0);
    }

    #[test]
    fn test_compression_stats_display() {
        let stats = CompressionStats {
            buffer_count: 5,
            total_original_size: 10_485_760,  // 10 MB
            total_compressed_size: 2_097_152, // 2 MB
            compression_ratio: 5.0,
            memory_saved: 8_388_608, // 8 MB
            algorithm: CompressionAlgorithm::Gzip,
        };

        let display = format!("{}", stats);
        assert!(display.contains("Algorithm: Gzip"));
        assert!(display.contains("Buffers: 5"));
        assert!(display.contains("10.00 MB"));
        assert!(display.contains("2.00 MB"));
        assert!(display.contains("5.00x"));
        assert!(display.contains("8.00 MB"));
    }

    #[test]
    fn test_decompression_size_mismatch() {
        // Create a buffer with wrong original size to test error handling
        let data = vec![1u8, 2, 3, 4];
        let mut buffer =
            CompressedBuffer::new(&data, CompressionAlgorithm::None, CompressionLevel::Default)
                .expect("Failed to create buffer");

        // Corrupt the original size
        buffer.original_size = 10; // Wrong size

        let result = buffer.decompress();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_compression_algorithm_equality() {
        assert_eq!(CompressionAlgorithm::Gzip, CompressionAlgorithm::Gzip);
        assert_ne!(CompressionAlgorithm::Gzip, CompressionAlgorithm::Lz4);
        assert_ne!(CompressionAlgorithm::Lz4, CompressionAlgorithm::None);
    }

    #[test]
    fn test_compressed_array_accessors() {
        let array = Array2::<f32>::from_elem((10, 20), 42.0);

        let compressed =
            CompressedArray::from_array(&array, CompressionAlgorithm::Gzip, CompressionLevel::Best)
                .expect("Failed to create compressed array");

        assert!(compressed.compression_ratio() > 1.0);
        assert!(compressed.compressed_size() < compressed.original_size());
        assert_eq!(compressed.shape(), &array.raw_dim());
    }

    #[test]
    fn test_compression_with_empty_data() {
        let data: Vec<f64> = vec![];

        let buffer =
            CompressedBuffer::new(&data, CompressionAlgorithm::Gzip, CompressionLevel::Default)
                .expect("Failed to create buffer");

        assert_eq!(buffer.original_size(), 0);
        let decompressed = buffer.decompress().expect("Failed to decompress");
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_lz4_compression_level_clamping() {
        let data: Vec<u64> = vec![12345; 100];

        // LZ4 max level is 12, so 20 should be clamped to 12
        let buffer = CompressedBuffer::new(
            &data,
            CompressionAlgorithm::Lz4,
            CompressionLevel::Custom(20),
        )
        .expect("Failed to create buffer");

        let decompressed = buffer.decompress().expect("Failed to decompress");
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_adaptive_compression_small_data() {
        // Test with very small data (less than sample size)
        let small_data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let algorithm = AdaptiveCompression::choose_algorithm(&small_data);
        // Small data usually doesn't compress well
        assert!(matches!(
            algorithm,
            CompressionAlgorithm::None | CompressionAlgorithm::Lz4
        ));
    }

    #[test]
    fn test_compression_types() {
        // Test with different numeric types
        let u8_data: Vec<u8> = vec![255; 100];
        let u16_data: Vec<u16> = vec![65535; 100];
        let i64_data: Vec<i64> = vec![-1; 100];

        let _u8_buffer = CompressedBuffer::new(
            &u8_data,
            CompressionAlgorithm::Gzip,
            CompressionLevel::Default,
        )
        .expect("Failed with u8");
        let _u16_buffer =
            CompressedBuffer::new(&u16_data, CompressionAlgorithm::Lz4, CompressionLevel::Fast)
                .expect("Failed with u16");
        let _i64_buffer = CompressedBuffer::new(
            &i64_data,
            CompressionAlgorithm::None,
            CompressionLevel::Best,
        )
        .expect("Failed with i64");
    }
}
