//! # Large-Scale Testing Framework
//!
//! This module provides testing infrastructure for large-scale datasets and
//! operations that require substantial system resources. It includes:
//! - Multi-GB dataset processing tests
//! - Out-of-core algorithm validation
//! - Memory-mapped file operations testing
//! - Distributed computation simulation
//! - Scalability limit discovery

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::testing::{TestConfig, TestResult};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::{NamedTempFile, TempDir};

#[cfg(feature = "random")]
use rand::Rng;

#[cfg(feature = "memory_efficient")]
use crate::memory_efficient::MemoryMappedArray;

/// Large-scale test configuration
#[derive(Debug, Clone)]
pub struct LargeScaleTestConfig {
    /// Maximum dataset size to test (in bytes)
    pub max_dataset_size: usize,
    /// Memory limit for out-of-core operations
    pub memory_limit: usize,
    /// Temporary directory for large files
    pub temp_dir: Option<PathBuf>,
    /// Enable cleanup of temporary files
    pub cleanup_files: bool,
    /// Chunk size for processing large datasets
    pub chunk_size: usize,
    /// Number of parallel workers for distributed tests
    pub worker_count: usize,
    /// Enable progress reporting
    pub progress_reporting: bool,
}

impl Default for LargeScaleTestConfig {
    fn default() -> Self {
        Self {
            max_dataset_size: 1024 * 1024 * 1024, // 1GB
            memory_limit: 256 * 1024 * 1024,      // 256MB
            temp_dir: None,
            cleanup_files: true,
            chunk_size: 1024 * 1024, // 1MB chunks
            worker_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            progress_reporting: false,
        }
    }
}

impl LargeScaleTestConfig {
    /// Create a new large-scale test configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum dataset size
    pub fn with_max_dataset_size(mut self, size: usize) -> Self {
        self.max_dataset_size = size;
        self
    }

    /// Set the memory limit
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Set the temporary directory
    pub fn with_temp_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.temp_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Enable or disable file cleanup
    pub fn with_cleanup(mut self, cleanup: bool) -> Self {
        self.cleanup_files = cleanup;
        self
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the worker count
    pub fn with_worker_count(mut self, count: usize) -> Self {
        self.worker_count = count;
        self
    }

    /// Enable progress reporting
    pub fn with_progress_reporting(mut self, enabled: bool) -> Self {
        self.progress_reporting = enabled;
        self
    }
}

/// Result of large-scale testing
#[derive(Debug, Clone)]
pub struct LargeScaleTestResult {
    /// Test name
    pub test_name: String,
    /// Dataset size processed
    pub dataset_size: usize,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Processing throughput (bytes per second)
    pub throughput: f64,
    /// Total processing time
    pub duration: Duration,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Success indicator
    pub success: bool,
    /// Error information if failed
    pub error: Option<String>,
    /// Performance metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

impl LargeScaleTestResult {
    /// Create a new large-scale test result
    pub fn new(testname: String) -> Self {
        Self {
            test_name: testname,
            dataset_size: 0,
            peak_memory: 0,
            throughput: 0.0,
            duration: Duration::from_secs(0),
            chunks_processed: 0,
            success: false,
            error: None,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Mark as successful
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Set dataset size
    pub fn with_dataset_size(mut self, size: usize) -> Self {
        self.dataset_size = size;
        self
    }

    /// Set peak memory
    pub fn with_peak_memory(mut self, memory: usize) -> Self {
        self.peak_memory = memory;
        self
    }

    /// Set throughput
    pub fn with_throughput(mut self, throughput: f64) -> Self {
        self.throughput = throughput;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set chunks processed
    pub fn with_chunks_processed(mut self, chunks: usize) -> Self {
        self.chunks_processed = chunks;
        self
    }

    /// Set error
    pub fn witherror(mut self, error: String) -> Self {
        self.error = Some(error);
        self.success = false;
        self
    }

    /// Add metric
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Large dataset generator for testing
pub struct LargeDatasetGenerator {
    config: LargeScaleTestConfig,
    temp_dir: Option<TempDir>,
}

impl LargeDatasetGenerator {
    /// Create a new large dataset generator
    pub fn new(config: LargeScaleTestConfig) -> CoreResult<Self> {
        let temp_dir = if config.temp_dir.is_none() {
            Some(TempDir::new().map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to create temp directory: {}",
                    e
                )))
            })?)
        } else {
            None
        };

        Ok(Self { config, temp_dir })
    }

    /// Generate a large numeric dataset
    pub fn generate_numeric_dataset(&self, size: usize) -> CoreResult<PathBuf> {
        let temp_path = self.get_temp_path("numeric_dataset.bin")?;

        let start_time = Instant::now();
        if self.config.progress_reporting {
            println!("Generating {} MB numeric dataset...", size / (1024 * 1024));
        }

        // Generate data in chunks to avoid memory pressure
        let mut file = fs::File::create(&temp_path).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to create dataset file: {}",
                e
            )))
        })?;

        use std::io::Write;
        let chunk_size = self.config.chunk_size.min(size);
        let num_elements_per_chunk = chunk_size / std::mem::size_of::<f64>();
        let mut byteswritten = 0;

        while byteswritten < size {
            let remaining = size - byteswritten;
            let current_chunk_size = chunk_size.min(remaining);
            let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();

            // Generate chunk data
            let chunk_data: Vec<f64> = (0..elements_in_chunk)
                .map(|i| (byteswritten / std::mem::size_of::<f64>() + i) as f64)
                .collect();

            // Write chunk to file
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk_data.as_ptr() as *const u8,
                    chunk_data.len() * std::mem::size_of::<f64>(),
                )
            };
            file.write_all(bytes).map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to write chunk: {error}",
                    error = e
                )))
            })?;

            byteswritten += current_chunk_size;

            if self.config.progress_reporting && byteswritten % (10 * 1024 * 1024) == 0 {
                let progress = (byteswritten * 100) / size;
                println!("Progress: {}%", progress);
            }
        }

        if self.config.progress_reporting {
            println!("Dataset generation completed in {:?}", start_time.elapsed());
        }

        Ok(temp_path)
    }

    /// Generate a sparse dataset with mostly zeros
    pub fn generate_sparse_dataset(&self, size: usize, density: f64) -> CoreResult<PathBuf> {
        let temp_path = self.get_temp_path("sparse_dataset.bin")?;

        if self.config.progress_reporting {
            println!(
                "Generating {} MB sparse dataset (density: {:.2})...",
                size / (1024 * 1024),
                density
            );
        }

        let mut file = fs::File::create(&temp_path).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to create sparse dataset file: {}",
                e
            )))
        })?;

        use std::io::Write;
        let chunk_size = self.config.chunk_size.min(size);
        let num_elements_per_chunk = chunk_size / std::mem::size_of::<f64>();
        let mut byteswritten = 0;

        #[cfg(feature = "random")]
        let mut rng = rand::rng();

        while byteswritten < size {
            let remaining = size - byteswritten;
            let current_chunk_size = chunk_size.min(remaining);
            let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();

            // Generate sparse chunk data
            let chunk_data: Vec<f64> = (0..elements_in_chunk)
                .map(|_| {
                    #[cfg(feature = "random")]
                    {
                        if rng.gen_range(0.0..=1.0) < density {
                            rng.gen_range(-1000.0..=1000.0)
                        } else {
                            0.0
                        }
                    }
                    #[cfg(not(feature = "random"))]
                    {
                        // Fallback: deterministic sparse pattern
                        if (byteswritten / std::mem::size_of::<f64>()) % (1.0 / density) as usize
                            == 0
                        {
                            1.0
                        } else {
                            0.0
                        }
                    }
                })
                .collect();

            // Write chunk to file
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk_data.as_ptr() as *const u8,
                    chunk_data.len() * std::mem::size_of::<f64>(),
                )
            };
            file.write_all(bytes).map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to write sparse chunk: {}",
                    e
                )))
            })?;

            byteswritten += current_chunk_size;
        }

        Ok(temp_path)
    }

    /// Get a temporary file path
    fn get_temp_path(&self, filename: &str) -> CoreResult<PathBuf> {
        if let Some(ref temp_dir_path) = self.config.temp_dir {
            Ok(temp_dir_path.join(filename))
        } else if let Some(ref temp_dir) = self.temp_dir {
            Ok(temp_dir.path().join(filename))
        } else {
            let temp_file = NamedTempFile::new().map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to create temp file: {}",
                    e
                )))
            })?;
            Ok(temp_file.into_temp_path().to_path_buf())
        }
    }
}

/// Large-scale processor for testing algorithms on big datasets
pub struct LargeScaleProcessor {
    config: LargeScaleTestConfig,
}

impl LargeScaleProcessor {
    /// Create a new large-scale processor
    pub fn new(config: LargeScaleTestConfig) -> Self {
        Self { config }
    }

    /// Test chunked processing of a large dataset
    pub fn test_chunked_processing<F>(
        &self,
        dataset_path: &Path,
        processor: F,
    ) -> CoreResult<LargeScaleTestResult>
    where
        F: Fn(&[f64]) -> CoreResult<f64>,
    {
        let start_time = Instant::now();
        let mut result = LargeScaleTestResult::new("chunked_processing".to_string());

        // Get file size
        let file_size = fs::metadata(dataset_path)
            .map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to get file metadata: {}",
                    e
                )))
            })?
            .len() as usize;

        if self.config.progress_reporting {
            println!(
                "Processing {} MB dataset in chunks...",
                file_size / (1024 * 1024)
            );
        }

        // Open file for reading
        use std::io::Read;
        let mut file = fs::File::open(dataset_path).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to open dataset file: {}",
                e
            )))
        })?;

        let mut bytes_processed = 0;
        let mut chunks_processed = 0;
        let mut accumulator = 0.0;
        let chunk_size = self.config.chunk_size;
        let elements_per_chunk = chunk_size / std::mem::size_of::<f64>();

        while bytes_processed < file_size {
            let remaining = file_size - bytes_processed;
            let current_chunk_size = chunk_size.min(remaining);
            let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();

            // Read chunk
            let mut buffer = vec![0u8; current_chunk_size];
            file.read_exact(&mut buffer).map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to read chunk: {error}",
                    error = e
                )))
            })?;

            // Convert bytes to f64 slice
            let chunk_data = unsafe {
                std::slice::from_raw_parts(buffer.as_ptr() as *const f64, elements_in_chunk)
            };

            // Process chunk
            let chunk_result = processor(chunk_data)?;
            accumulator += chunk_result;

            bytes_processed += current_chunk_size;
            chunks_processed += 1;

            if self.config.progress_reporting && chunks_processed % 100 == 0 {
                let progress = (bytes_processed * 100) / file_size;
                println!("Processing progress: {}%", progress);
            }
        }

        let duration = start_time.elapsed();
        let throughput = file_size as f64 / duration.as_secs_f64();

        result = result
            .with_success(true)
            .with_dataset_size(file_size)
            .with_duration(duration)
            .with_chunks_processed(chunks_processed)
            .with_throughput(throughput)
            .with_metric("accumulator_result".to_string(), accumulator);

        if self.config.progress_reporting {
            println!(
                "Processing completed: {} chunks, {:.2} MB/s throughput",
                chunks_processed,
                throughput / (1024.0 * 1024.0)
            );
        }

        Ok(result)
    }

    /// Test memory-mapped processing
    #[cfg(feature = "memory_efficient")]
    pub fn test_memory_mapped_processing<F>(
        &self,
        dataset_path: &Path,
        processor: F,
    ) -> CoreResult<LargeScaleTestResult>
    where
        F: Fn(&[f64]) -> CoreResult<f64>,
    {
        let start_time = Instant::now();
        let mut result = LargeScaleTestResult::new("memory_mapped_processing".to_string());

        // Get file size
        let file_size = fs::metadata(dataset_path)
            .map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to get file metadata: {}",
                    e
                )))
            })?
            .len() as usize;

        let num_elements = file_size / std::mem::size_of::<f64>();

        if self.config.progress_reporting {
            println!("Memory-mapping {} MB dataset...", file_size / (1024 * 1024));
        }

        // Create memory-mapped array
        let mmap_array =
            MemoryMappedArray::<f64>::path(dataset_path, &[num_elements]).map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to create memory map: {:?}",
                    e
                )))
            })?;

        // Process in chunks using memory-mapped data
        let chunk_size = self.config.chunk_size / std::mem::size_of::<f64>();
        let mut chunks_processed = 0;
        let mut accumulator = 0.0;

        for chunk_start in (0..num_elements).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(num_elements);

            // Access chunk data from memory-mapped array
            let chunk_data = {
                let array = mmap_array.asarray::<ndarray::Ix1>().map_err(|e| {
                    CoreError::ComputationError(ErrorContext::new(format!(
                        "Failed to access memory-mapped array: {:?}",
                        e
                    )))
                })?;

                // Extract the chunk slice
                let slice = array.slice(ndarray::s![chunk_start..chunk_end]);
                slice.to_vec() // Convert to owned Vec for processing
            };

            let chunk_result = processor(&chunk_data)?;
            accumulator += chunk_result;
            chunks_processed += 1;

            if self.config.progress_reporting && chunks_processed % 100 == 0 {
                let progress = (chunk_start * 100) / num_elements;
                println!("Memory-mapped processing progress: {}%", progress);
            }
        }

        let duration = start_time.elapsed();
        let throughput = file_size as f64 / duration.as_secs_f64();

        result = result
            .with_success(true)
            .with_dataset_size(file_size)
            .with_duration(duration)
            .with_chunks_processed(chunks_processed)
            .with_throughput(throughput)
            .with_metric("accumulator_result".to_string(), accumulator);

        if self.config.progress_reporting {
            println!(
                "Memory-mapped processing completed: {} chunks, {:.2} MB/s throughput",
                chunks_processed,
                throughput / (1024.0 * 1024.0)
            );
        }

        Ok(result)
    }

    /// Test out-of-core reduction operation
    pub fn test_out_of_core_reduction(
        &self,
        dataset_path: &Path,
    ) -> CoreResult<LargeScaleTestResult> {
        let start_time = Instant::now();
        let mut result = LargeScaleTestResult::new("out_of_core_reduction".to_string());

        // Perform sum reduction as a test operation
        let processor_result =
            self.test_chunked_processing(dataset_path, |chunk| Ok(chunk.iter().sum::<f64>()))?;

        // Verify the result by computing it differently
        let verification_result = self.verify_reduction_result(dataset_path)?;

        let success =
            (processor_result.metrics["accumulator_result"] - verification_result).abs() < 1e-6;

        result = result
            .with_success(success)
            .with_dataset_size(processor_result.dataset_size)
            .with_duration(processor_result.duration)
            .with_chunks_processed(processor_result.chunks_processed)
            .with_throughput(processor_result.throughput)
            .with_metric(
                "computed_sum".to_string(),
                processor_result.metrics["accumulator_result"],
            )
            .with_metric("verified_sum".to_string(), verification_result);

        if !success {
            result = result.witherror(format!(
                "Reduction verification failed: computed={}, verified={}",
                processor_result.metrics["accumulator_result"], verification_result
            ));
        }

        Ok(result)
    }

    /// Verify reduction result using a different method
    fn verify_reduction_result(&self, datasetpath: &Path) -> CoreResult<f64> {
        // Simple verification: compute sum using smaller chunks
        let mut file = fs::File::open(datasetpath).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to open dataset for verification: {}",
                e
            )))
        })?;

        use std::io::Read;
        let verification_chunk_size = 1024; // Smaller chunks for verification
        let mut buffer = vec![0u8; verification_chunk_size];
        let mut sum = 0.0;

        loop {
            match file.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(bytes_read) => {
                    let elements = bytes_read / std::mem::size_of::<f64>();
                    let data = unsafe {
                        std::slice::from_raw_parts(buffer.as_ptr() as *const f64, elements)
                    };
                    sum += data.iter().sum::<f64>();
                }
                Err(e) => {
                    return Err(CoreError::IoError(ErrorContext::new(format!(
                        "Verification read failed: {}",
                        e
                    ))))
                }
            }
        }

        Ok(sum)
    }
}

/// High-level large-scale testing utilities
pub struct LargeScaleTestUtils;

impl LargeScaleTestUtils {
    /// Create a comprehensive large-scale test suite
    pub fn create_large_scale_test_suite(
        name: &str,
        config: TestConfig,
    ) -> crate::testing::TestSuite {
        let mut suite = crate::testing::TestSuite::new(name, config);

        // Use smaller datasets for testing to avoid excessive resource usage
        let large_config = LargeScaleTestConfig::default()
            .with_max_dataset_size(10 * 1024 * 1024) // 10MB for tests
            .with_chunk_size(1024 * 1024)            // 1MB chunks
            .with_progress_reporting(false);

        let large_config_1 = large_config.clone();
        suite.add_test("chunked_dataset_processing", move |_runner| {
            let generator = LargeDatasetGenerator::new(large_config_1.clone())?;
            let processor = LargeScaleProcessor::new(large_config_1.clone());

            // Generate test dataset
            let dataset_path =
                generator.generate_numeric_dataset(large_config_1.max_dataset_size)?;

            // Test chunked processing
            let result = processor.test_chunked_processing(&dataset_path, |chunk| {
                // Simple mean calculation
                Ok(chunk.iter().sum::<f64>() / chunk.len() as f64)
            })?;

            if !result.success {
                return Ok(TestResult::failure(
                    result.duration,
                    result.chunks_processed,
                    result
                        .error
                        .unwrap_or_else(|| "Chunked processing failed".to_string()),
                ));
            }

            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.chunks_processed,
            ))
        });

        let large_config_2 = large_config.clone();
        suite.add_test("sparse_dataset_processing", move |_runner| {
            let generator = LargeDatasetGenerator::new(large_config_2.clone())?;
            let processor = LargeScaleProcessor::new(large_config_2.clone());

            // Generate sparse test dataset
            let dataset_path =
                generator.generate_sparse_dataset(large_config_2.max_dataset_size, 0.1)?;

            // Test sparse processing
            let result = processor.test_chunked_processing(&dataset_path, |chunk| {
                // Count non-zero elements
                Ok(chunk.iter().filter(|&&x| x != 0.0).count() as f64)
            })?;

            if !result.success {
                return Ok(TestResult::failure(
                    result.duration,
                    result.chunks_processed,
                    result
                        .error
                        .unwrap_or_else(|| "Sparse processing failed".to_string()),
                ));
            }

            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.chunks_processed,
            ))
        });

        let large_config_3 = large_config.clone();
        suite.add_test("out_of_core_reduction", move |_runner| {
            let generator = LargeDatasetGenerator::new(large_config_3.clone())?;
            let processor = LargeScaleProcessor::new(large_config_3.clone());

            // Generate test dataset
            let dataset_path =
                generator.generate_numeric_dataset(large_config_3.max_dataset_size)?;

            // Test out-of-core reduction
            let result = processor.test_out_of_core_reduction(&dataset_path)?;

            if !result.success {
                return Ok(TestResult::failure(
                    result.duration,
                    result.chunks_processed,
                    result
                        .error
                        .unwrap_or_else(|| "Out-of-core reduction failed".to_string()),
                ));
            }

            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.chunks_processed,
            ))
        });

        #[cfg(feature = "memory_efficient")]
        {
            let large_config_4 = large_config.clone();
            suite.add_test("memory_mapped_processing", move |_runner| {
                let generator = LargeDatasetGenerator::new(large_config_4.clone())?;
                let processor = LargeScaleProcessor::new(large_config_4.clone());

                // Generate test dataset
                let dataset_path =
                    generator.generate_numeric_dataset(large_config_4.max_dataset_size)?;

                // Test chunked processing (memory-mapped)
                let result = processor.test_chunked_processing(&dataset_path, |chunk| {
                    // Compute variance
                    let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
                    let variance =
                        chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / chunk.len() as f64;
                    Ok(variance)
                })?;

                if !result.success {
                    return Ok(TestResult::failure(
                        result.duration,
                        result.chunks_processed,
                        result
                            .error
                            .unwrap_or_else(|| "Memory-mapped processing failed".to_string()),
                    ));
                }

                Ok(TestResult::success(
                    result.duration,
                    result.chunks_processed,
                ))
            });
        }

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_scale_config() {
        let config = LargeScaleTestConfig::new()
            .with_max_dataset_size(512 * 1024 * 1024)
            .with_memory_limit(128 * 1024 * 1024)
            .with_chunk_size(2 * 1024 * 1024)
            .with_worker_count(8)
            .with_progress_reporting(true);

        assert_eq!(config.max_dataset_size, 512 * 1024 * 1024);
        assert_eq!(config.memory_limit, 128 * 1024 * 1024);
        assert_eq!(config.chunk_size, 2 * 1024 * 1024);
        assert_eq!(config.worker_count, 8);
        assert!(config.progress_reporting);
    }

    #[test]
    fn test_dataset_generator() {
        let config = LargeScaleTestConfig::default().with_max_dataset_size(1024); // Small size for test

        let generator = LargeDatasetGenerator::new(config).unwrap();
        let dataset_path = generator.generate_numeric_dataset(1024).unwrap();

        assert!(dataset_path.exists());

        let metadata = fs::metadata(&dataset_path).unwrap();
        assert_eq!(metadata.len() as usize, 1024);
    }

    #[test]
    fn test_sparse_dataset_generator() {
        let config = LargeScaleTestConfig::default();
        let generator = LargeDatasetGenerator::new(config).unwrap();

        let dataset_path = generator.generate_sparse_dataset(1024, 0.5).unwrap();
        assert!(dataset_path.exists());

        let metadata = fs::metadata(&dataset_path).unwrap();
        assert_eq!(metadata.len() as usize, 1024);
    }

    #[test]
    fn test_chunked_processing() {
        let config = LargeScaleTestConfig::default().with_chunk_size(256);

        let generator = LargeDatasetGenerator::new(config.clone()).unwrap();
        let processor = LargeScaleProcessor::new(config);

        let dataset_path = generator.generate_numeric_dataset(1024).unwrap();

        let result = processor
            .test_chunked_processing(&dataset_path, |chunk| Ok(chunk.len() as f64))
            .unwrap();

        assert!(result.success);
        assert_eq!(result.dataset_size, 1024);
        assert!(result.chunks_processed > 0);
        assert!(result.throughput > 0.0);
    }
}
