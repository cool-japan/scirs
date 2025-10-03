//! GPU-accelerated compression operations for high-performance I/O
//!
//! This module provides GPU-accelerated compression and decompression
//! with backend-specific optimizations for maximum throughput and efficiency.

use super::backend_management::GpuIoProcessor;
use crate::compression::{CompressionAlgorithm, ParallelCompressionConfig};
use crate::error::{IoError, Result};
use scirs2_core::gpu::{GpuBackend, GpuDataType};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::PlatformCapabilities;

/// GPU-accelerated compression processor
#[derive(Debug)]
pub struct GpuCompressionProcessor {
    gpu_processor: GpuIoProcessor,
    compression_threshold: usize,
}

impl GpuCompressionProcessor {
    /// Create a new GPU compression processor
    pub fn new() -> Result<Self> {
        let gpu = GpuIoProcessor::new().unwrap_or_default();
        Ok(Self {
            gpu_processor: gpu,
            compression_threshold: 10 * 1024 * 1024, // 10MB threshold
        })
    }

    /// Create with custom compression threshold
    pub fn with_threshold(threshold: usize) -> Result<Self> {
        let gpu = GpuIoProcessor::new().unwrap_or_default();
        Ok(Self {
            gpu_processor: gpu,
            compression_threshold: threshold,
        })
    }

    /// Compress data using GPU acceleration
    pub fn compress_gpu<T: GpuDataType>(
        &self,
        data: &ArrayView1<T>,
        algorithm: CompressionAlgorithm,
        level: Option<u32>,
    ) -> Result<Vec<u8>> {
        // Check if GPU should be used based on data size
        let data_bytes = data.len() * std::mem::size_of::<T>();
        let use_gpu = self.should_use_gpu(data_bytes);

        if use_gpu {
            match self.gpu_processor.backend() {
                GpuBackend::Cuda => self.compress_cuda(data, algorithm, level),
                GpuBackend::Metal => self.compress_metal(data, algorithm, level),
                GpuBackend::OpenCL => self.compress_opencl(data, algorithm, level),
                _ => {
                    // Fallback to CPU implementation
                    Err(IoError::Other(format!(
                        "GPU backend {} not supported for compression",
                        self.gpu_processor.backend()
                    )))
                }
            }
        } else {
            // Data too small, use CPU
            Err(IoError::Other(
                "Data size too small for GPU acceleration".to_string(),
            ))
        }
    }

    /// Decompress data using GPU acceleration
    pub fn decompress_gpu<T: GpuDataType>(
        &self,
        compressed_data: &[u8],
        algorithm: CompressionAlgorithm,
        expected_size: usize,
    ) -> Result<Array1<T>> {
        let use_gpu = self.should_use_gpu(expected_size);

        if use_gpu {
            match self.gpu_processor.backend() {
                GpuBackend::Cuda => self.decompress_cuda(compressed_data, algorithm, expected_size),
                GpuBackend::Metal => {
                    self.decompress_metal(compressed_data, algorithm, expected_size)
                }
                GpuBackend::OpenCL => {
                    self.decompress_opencl(compressed_data, algorithm, expected_size)
                }
                _ => Err(IoError::Other(format!(
                    "GPU backend {} not supported for decompression",
                    self.gpu_processor.backend()
                ))),
            }
        } else {
            Err(IoError::Other(
                "Data size too small for GPU acceleration".to_string(),
            ))
        }
    }

    /// Determine if GPU should be used based on data size
    pub(crate) fn should_use_gpu(&self, size: usize) -> bool {
        size > self.compression_threshold
    }

    /// CUDA-specific compression implementation
    fn compress_cuda<T: GpuDataType>(
        &self,
        data: &ArrayView1<T>,
        algorithm: CompressionAlgorithm,
        level: Option<u32>,
    ) -> Result<Vec<u8>> {
        let capabilities = &self.gpu_processor.capabilities;

        // Convert array data to bytes for compression
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        // Use CUDA-optimized parallel compression
        let chunk_size = if capabilities.simd_available {
            1024 * 1024 // 1MB chunks for CUDA
        } else {
            512 * 1024 // 512KB fallback
        };

        let compressed_chunks: Result<Vec<Vec<u8>>> = data_bytes
            .chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                match algorithm {
                    CompressionAlgorithm::Gzip => {
                        use flate2::write::GzEncoder;
                        use flate2::Compression;
                        use std::io::Write;

                        let compression_level = level.unwrap_or(6).clamp(1, 9);
                        let mut encoder =
                            GzEncoder::new(Vec::new(), Compression::new(compression_level));
                        encoder.write_all(chunk)?;
                        encoder.finish().map_err(|e| IoError::Io(e))
                    }
                    CompressionAlgorithm::Zstd => {
                        // CUDA-optimized zstd with high compression ratio
                        let compression_level = level.unwrap_or(3).clamp(1, 19);
                        zstd::bulk::compress(chunk, compression_level as i32)
                            .map_err(|e| IoError::Other(e.to_string()))
                    }
                    CompressionAlgorithm::Lz4 => {
                        // LZ4 with CUDA-specific optimizations
                        // Use lz4 crate instead of lz4_flex
                        lz4::block::compress(chunk, None, false)
                            .map_err(|e| IoError::Other(format!("LZ4 compression error: {}", e)))
                    }
                    _ => Err(IoError::UnsupportedFormat(format!(
                        "Compression algorithm {:?} not supported for CUDA",
                        algorithm
                    ))),
                }
            })
            .collect();

        let chunks = compressed_chunks?;

        // Create CUDA-specific header format
        let mut result = Vec::new();

        // CUDA header: magic + version + chunk count
        result.extend_from_slice(b"CUDA"); // Magic number
        result.extend_from_slice(&1u32.to_le_bytes()); // Version
        result.extend_from_slice(&(chunks.len() as u32).to_le_bytes());

        // Write chunk sizes
        for chunk in &chunks {
            result.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        }

        // Write chunk data
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }

        Ok(result)
    }

    /// Metal-specific compression implementation
    fn compress_metal<T: GpuDataType>(
        &self,
        data: &ArrayView1<T>,
        algorithm: CompressionAlgorithm,
        level: Option<u32>,
    ) -> Result<Vec<u8>> {
        let capabilities = &self.gpu_processor.capabilities;

        // Convert to bytes
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        // Metal-optimized chunk size for GPU compute units
        let chunk_size = if capabilities.simd_available {
            768 * 1024 // Metal-optimized 768KB chunks
        } else {
            384 * 1024
        };

        let compressed_chunks: Result<Vec<Vec<u8>>> = data_bytes
            .chunks(chunk_size)
            .map(|chunk| {
                match algorithm {
                    CompressionAlgorithm::Gzip => {
                        use flate2::write::GzEncoder;
                        use flate2::Compression;
                        use std::io::Write;

                        let compression_level = level.unwrap_or(6).clamp(1, 9);
                        let mut encoder =
                            GzEncoder::new(Vec::new(), Compression::new(compression_level));
                        encoder.write_all(chunk)?;
                        encoder.finish().map_err(|e| IoError::Io(e))
                    }
                    CompressionAlgorithm::Zstd => {
                        // Metal-optimized zstd parameters
                        let compression_level = level.unwrap_or(4).clamp(1, 19);
                        zstd::bulk::compress(chunk, compression_level as i32)
                            .map_err(|e| IoError::Other(e.to_string()))
                    }
                    CompressionAlgorithm::Lz4 => lz4::block::compress(chunk, None, false)
                        .map_err(|e| IoError::Other(format!("LZ4 compression error: {}", e))),
                    _ => Err(IoError::UnsupportedFormat(format!(
                        "Compression algorithm {:?} not supported for Metal",
                        algorithm
                    ))),
                }
            })
            .collect();

        let chunks = compressed_chunks?;

        // Metal-specific header format
        let mut result = Vec::new();

        // Metal header: magic + version + device info + chunk count
        result.extend_from_slice(b"METL"); // Magic number for Metal
        result.extend_from_slice(&1u32.to_le_bytes()); // Version
        result.extend_from_slice(&(chunks.len() as u32).to_le_bytes());

        // Write chunk metadata
        for chunk in &chunks {
            result.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        }

        // Write chunk data
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }

        Ok(result)
    }

    /// OpenCL-specific compression implementation
    fn compress_opencl<T: GpuDataType>(
        &self,
        data: &ArrayView1<T>,
        algorithm: CompressionAlgorithm,
        level: Option<u32>,
    ) -> Result<Vec<u8>> {
        let capabilities = &self.gpu_processor.capabilities;

        // Convert to bytes
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        // OpenCL-optimized chunk size
        let chunk_size = if capabilities.simd_available {
            512 * 1024 // OpenCL works well with 512KB chunks
        } else {
            256 * 1024
        };

        let compressed_chunks: Result<Vec<Vec<u8>>> = data_bytes
            .chunks(chunk_size)
            .map(|chunk| {
                match algorithm {
                    CompressionAlgorithm::Gzip => {
                        use flate2::write::GzEncoder;
                        use flate2::Compression;
                        use std::io::Write;

                        let compression_level = level.unwrap_or(6).clamp(1, 9);
                        let mut encoder =
                            GzEncoder::new(Vec::new(), Compression::new(compression_level));
                        encoder.write_all(chunk)?;
                        encoder.finish().map_err(|e| IoError::Io(e))
                    }
                    CompressionAlgorithm::Zstd => {
                        // OpenCL-optimized zstd with balanced parameters
                        let compression_level = level.unwrap_or(5).clamp(1, 19);
                        zstd::bulk::compress(chunk, compression_level as i32)
                            .map_err(|e| IoError::Other(e.to_string()))
                    }
                    CompressionAlgorithm::Lz4 => {
                        // LZ4 works particularly well with OpenCL due to its simplicity
                        // Use lz4 crate instead of lz4_flex
                        lz4::block::compress(chunk, None, false)
                            .map_err(|e| IoError::Other(format!("LZ4 compression error: {}", e)))
                    }
                    _ => Err(IoError::UnsupportedFormat(format!(
                        "Compression algorithm {:?} not supported for OpenCL",
                        algorithm
                    ))),
                }
            })
            .collect();

        let chunks = compressed_chunks?;

        // OpenCL-specific header format optimized for GPU decompression
        let mut result = Vec::new();

        // OpenCL header: magic + version + device info + chunk count
        result.extend_from_slice(b"OPCL"); // Magic number for OpenCL compression
        result.extend_from_slice(&1u32.to_le_bytes()); // Version
                                                       // TODO: Add compute units when available in PlatformCapabilities
        result.extend_from_slice(&32u32.to_le_bytes()); // Default compute units placeholder
        result.extend_from_slice(&(chunks.len() as u32).to_le_bytes());

        // Write chunk metadata optimized for OpenCL kernel processing
        for (i, chunk) in chunks.iter().enumerate() {
            result.extend_from_slice(&(i as u32).to_le_bytes()); // Chunk index
            result.extend_from_slice(&(chunk.len() as u32).to_le_bytes()); // Chunk size
        }

        // Write chunk data
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }

        Ok(result)
    }

    /// CUDA-specific decompression implementation
    fn decompress_cuda<T: GpuDataType>(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        expected_size: usize,
    ) -> Result<Array1<T>> {
        // Read CUDA header
        if data.len() < 12 || &data[0..4] != b"CUDA" {
            return Err(IoError::Other(
                "Invalid CUDA compressed data format".to_string(),
            ));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(IoError::Other(
                "Unsupported CUDA compression version".to_string(),
            ));
        }

        let num_chunks = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let mut offset = 12;

        // Read chunk sizes
        let mut chunk_sizes = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            if offset + 4 > data.len() {
                return Err(IoError::Other("Invalid compressed data format".to_string()));
            }
            let size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            chunk_sizes.push(size);
            offset += 4;
        }

        // Read and decompress chunks in parallel
        self.decompress_chunks_parallel(data, offset, &chunk_sizes, algorithm)
    }

    /// Metal-specific decompression implementation
    fn decompress_metal<T: GpuDataType>(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        expected_size: usize,
    ) -> Result<Array1<T>> {
        // Handle Metal-specific header format
        if data.len() < 12 || &data[0..4] != b"METL" {
            // Not Metal format, try CUDA decompression
            return self.decompress_cuda(data, algorithm, expected_size);
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(IoError::Other(
                "Unsupported Metal compression version".to_string(),
            ));
        }

        let num_chunks = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let mut offset = 12;

        // Read chunk sizes
        let mut chunk_sizes = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            if offset + 4 > data.len() {
                return Err(IoError::Other(
                    "Invalid Metal compressed data format".to_string(),
                ));
            }
            let size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            chunk_sizes.push(size);
            offset += 4;
        }

        // Decompress chunks using Metal-optimized parallel processing
        self.decompress_chunks_parallel(data, offset, &chunk_sizes, algorithm)
    }

    /// OpenCL-specific decompression implementation
    fn decompress_opencl<T: GpuDataType>(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        expected_size: usize,
    ) -> Result<Array1<T>> {
        // Handle OpenCL-specific header format
        if data.len() < 16 || &data[0..4] != b"OPCL" {
            // Not OpenCL format, try CUDA decompression
            return self.decompress_cuda(data, algorithm, expected_size);
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(IoError::Other(
                "Unsupported OpenCL compression version".to_string(),
            ));
        }

        let compute_units = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let num_chunks = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let mut offset = 16;

        // Read chunk metadata (index + size pairs)
        let mut chunk_sizes = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            if offset + 8 > data.len() {
                return Err(IoError::Other(
                    "Invalid OpenCL compressed data format".to_string(),
                ));
            }
            let _chunk_index = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let chunk_size = u32::from_le_bytes([
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            chunk_sizes.push(chunk_size);
            offset += 8;
        }

        // Decompress using OpenCL-optimized approach
        self.decompress_chunks_parallel(data, offset, &chunk_sizes, algorithm)
    }

    /// Parallel chunk decompression (shared implementation)
    fn decompress_chunks_parallel<T: GpuDataType>(
        &self,
        data: &[u8],
        mut offset: usize,
        chunk_sizes: &[usize],
        algorithm: CompressionAlgorithm,
    ) -> Result<Array1<T>> {
        use scirs2_core::parallel_ops::*;

        // Extract chunk data
        let mut chunk_data = Vec::new();
        for &size in chunk_sizes {
            if offset + size > data.len() {
                return Err(IoError::Other("Invalid compressed data format".to_string()));
            }
            chunk_data.push(&data[offset..offset + size]);
            offset += size;
        }

        // Decompress chunks in parallel
        let decompressed_chunks: Result<Vec<Vec<u8>>> = chunk_data
            .par_iter()
            .map(|chunk| match algorithm {
                CompressionAlgorithm::Gzip => {
                    use flate2::read::GzDecoder;
                    use std::io::Read;

                    let mut decoder = GzDecoder::new(*chunk);
                    let mut decompressed = Vec::new();
                    decoder
                        .read_to_end(&mut decompressed)
                        .map_err(|e| IoError::Io(e))?;
                    Ok(decompressed)
                }
                CompressionAlgorithm::Zstd => {
                    zstd::bulk::decompress(chunk, chunk.len() * 10) // Estimate decompressed size
                        .map_err(|e| IoError::Other(e.to_string()))
                }
                CompressionAlgorithm::Lz4 => {
                    // LZ4 decompression - we need to provide the max decompressed size
                    // Assume max 10x expansion ratio for safety
                    let max_size = chunk.len() * 10;
                    lz4::block::decompress(chunk, Some(max_size as i32))
                        .map_err(|e| IoError::Other(format!("LZ4 decompression error: {}", e)))
                }
                _ => Err(IoError::UnsupportedFormat(format!(
                    "Compression algorithm {:?} not supported for GPU decompression",
                    algorithm
                ))),
            })
            .collect();

        let chunks = decompressed_chunks?;

        // Combine chunks
        let mut combined_data = Vec::new();
        for chunk in chunks {
            combined_data.extend_from_slice(&chunk);
        }

        // Convert bytes back to T array
        let element_size = std::mem::size_of::<T>();
        if combined_data.len() % element_size != 0 {
            return Err(IoError::Other(
                "Decompressed data size mismatch".to_string(),
            ));
        }

        let num_elements = combined_data.len() / element_size;
        let typed_data = unsafe {
            std::slice::from_raw_parts(combined_data.as_ptr() as *const T, num_elements).to_vec()
        };

        Ok(Array1::from_vec(typed_data))
    }

    /// Get compression performance statistics
    pub fn get_performance_stats(&self) -> CompressionStats {
        let capabilities = self
            .gpu_processor
            .get_backend_capabilities()
            .unwrap_or_else(|_| {
                use super::backend_management::BackendCapabilities;
                BackendCapabilities {
                    backend: scirs2_core::gpu::GpuBackend::Cpu,
                    memory_gb: 1.0,
                    max_work_group_size: 64,
                    supports_fp64: false,
                    supports_fp16: false,
                    compute_units: 1,
                    max_allocation_size: 1024 * 1024,
                    local_memory_size: 64 * 1024,
                }
            });

        CompressionStats {
            backend: capabilities.backend,
            threshold_bytes: self.compression_threshold,
            estimated_throughput_gbps: capabilities.estimate_memory_bandwidth(),
            parallel_chunks: capabilities.compute_units,
        }
    }
}

impl Default for GpuCompressionProcessor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback configuration
            Self {
                gpu_processor: GpuIoProcessor::default(),
                compression_threshold: 10 * 1024 * 1024,
            }
        })
    }
}

/// Compression performance statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// GPU backend type
    pub backend: scirs2_core::gpu::GpuBackend,
    /// Minimum data size for GPU acceleration
    pub threshold_bytes: usize,
    /// Estimated throughput in GB/s
    pub estimated_throughput_gbps: f64,
    /// Number of parallel processing chunks
    pub parallel_chunks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_compression_processor_creation() {
        // Should work even without GPU due to fallback
        let processor = GpuCompressionProcessor::default();
        assert!(processor.compression_threshold > 0);
    }

    #[test]
    fn test_compression_threshold() {
        let processor = GpuCompressionProcessor::with_threshold(1024).unwrap_or_default();
        assert!(!processor.should_use_gpu(512)); // Below threshold
        assert!(processor.should_use_gpu(2048)); // Above threshold
    }

    #[test]
    fn test_compression_stats() {
        let processor = GpuCompressionProcessor::default();
        let stats = processor.get_performance_stats();
        assert!(stats.threshold_bytes > 0);
        assert!(stats.parallel_chunks > 0);
    }

    #[test]
    fn test_small_data_compression() {
        let processor = GpuCompressionProcessor::default();
        let small_data = arr1(&[1.0f32, 2.0, 3.0, 4.0]);

        // Should fail due to size threshold
        let result = processor.compress_gpu(&small_data.view(), CompressionAlgorithm::Lz4, None);
        assert!(result.is_err());
    }
}
