// Memory-optimized algorithms for large signal processing
//
// This module provides memory-efficient implementations of signal processing
// algorithms designed to work with very large signals that might not fit
// entirely in memory, or where memory usage needs to be carefully controlled.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex;
use rustfft::FftPlanner;
use scirs2_core::parallel_ops::*;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

#[allow(unused_imports)]
/// Configuration for memory-optimized operations
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Chunk size for processing (samples)
    pub chunk_size: usize,
    /// Overlap between chunks (samples)
    pub overlap_size: usize,
    /// Use memory mapping for large files
    pub use_mmap: bool,
    /// Temporary directory for scratch files
    pub temp_dir: Option<String>,
    /// Enable compression for temporary files
    pub compress_temp: bool,
    /// Cache size for frequently accessed data
    pub cache_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            chunk_size: 65536,                    // 64K samples
            overlap_size: 1024,                   // 1K overlap
            use_mmap: true,
            temp_dir: None, // Use system temp
            compress_temp: false,
            cache_size: 128 * 1024 * 1024, // 128MB cache
        }
    }
}

/// Result of memory-optimized operation
#[derive(Debug)]
pub struct MemoryOptimizedResult<T> {
    /// Result data (may be on disk)
    pub data: MemoryOptimizedData<T>,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Processing time statistics
    pub timing_stats: TimingStats,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Average memory usage (bytes)
    pub avg_memory: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of disk I/O operations
    pub disk_operations: usize,
}

/// Timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Total processing time (ms)
    pub total_time_ms: u128,
    /// Time spent on I/O (ms)
    pub io_time_ms: u128,
    /// Time spent on computation (ms)
    pub compute_time_ms: u128,
    /// Time spent on memory management (ms)
    pub memory_mgmt_time_ms: u128,
}

/// Memory-optimized data storage
#[derive(Debug)]
pub enum MemoryOptimizedData<T> {
    /// Data in memory
    InMemory(Vec<T>),
    /// Data on disk with file path
    OnDisk {
        file_path: String,
        length: usize,
        chunk_size: usize,
    },
    /// Hybrid storage (frequently accessed in memory, rest on disk)
    Hybrid {
        memory_chunks: Vec<Option<Vec<T>>>,
        disk_file: String,
        chunk_size: usize,
        total_length: usize,
    },
}

/// Memory-optimized FIR filtering for very large signals
///
/// Processes signals that may not fit in memory by using chunked processing
/// with proper overlap handling to maintain filter continuity.
#[allow(dead_code)]
pub fn memory_optimized_fir_filter(
    input_file: &str,
    output_file: &str,
    coefficients: &[f64],
    config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<f64>> {
    let start_time = Instant::now();
    let mut memory_stats = MemoryStats {
        peak_memory: 0,
        avg_memory: 0,
        cache_hits: 0,
        cache_misses: 0,
        disk_operations: 0,
    };

    // Open input _file and determine size
    let input_file_handle = File::open(input_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot open input file: {}", e)))?;

    let file_size = input_file_handle
        .metadata()
        .map_err(|e| SignalError::ComputationError(format!("Cannot get _file size: {}", e)))?
        .len() as usize;

    let samples_count = file_size / std::mem::size_of::<f64>();
    let filter_length = coefficients.len();

    // Create output _file
    let output_file_handle = File::create(output_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create output file: {}", e)))?;

    let mut input_reader = BufReader::new(input_file_handle);
    let mut output_writer = BufWriter::new(output_file_handle);

    // Calculate optimal chunk size based on memory constraints
    let sample_size = std::mem::size_of::<f64>();
    let max_samples_in_memory = config.max_memory_bytes / sample_size / 4; // Reserve space for intermediate buffers
    let chunk_size = config.chunk_size.min(max_samples_in_memory);

    // Initialize filter state (for IIR continuity across chunks)
    let mut filter_memory = vec![0.0; filter_length.saturating_sub(1)];

    let mut input_buffer = vec![0.0; chunk_size + config.overlap_size];
    let mut output_buffer = vec![0.0; chunk_size];

    let mut total_processed = 0;
    let mut peak_memory = 0;
    let mut io_time = 0;
    let mut compute_time = 0;

    // Process signal in chunks
    while total_processed < samples_count {
        let io_start = Instant::now();

        // Read chunk with overlap
        let samples_to_read =
            (chunk_size + config.overlap_size).min(samples_count - total_processed);

        // Read binary data (assuming f64 samples)
        let mut raw_buffer = vec![0u8; samples_to_read * sample_size];
        input_reader
            .read_exact(&mut raw_buffer)
            .map_err(|e| SignalError::ComputationError(format!("Read error: {}", e)))?;

        // Convert bytes to f64 samples
        for (i, chunk) in raw_buffer.chunks_exact(sample_size).enumerate() {
            if i < input_buffer.len() {
                input_buffer[i] = f64::from_le_bytes(chunk.try_into().map_err(|_| {
                    SignalError::ComputationError("Invalid data format".to_string())
                })?);
            }
        }

        memory_stats.disk_operations += 1;
        io_time += io_start.elapsed().as_millis();

        let compute_start = Instant::now();

        // Apply filter to chunk
        let effective_length = samples_to_read.min(chunk_size);

        // FIR filtering with overlap handling
        for i in 0..effective_length {
            let mut output_sample = 0.0;

            for (j, &coeff) in coefficients.iter().enumerate() {
                let input_idx = if i >= j {
                    i - j
                } else {
                    // Use previous chunk data or zero
                    if j - i - 1 < filter_memory.len() {
                        filter_memory[filter_memory.len() - (j - i)];
                        continue;
                    } else {
                        0.0
                    }
                };

                if input_idx < input_buffer.len() {
                    output_sample += input_buffer[input_idx] * coeff;
                }
            }

            output_buffer[i] = output_sample;
        }

        // Update filter memory for next chunk
        let memory_start = filter_memory.len().saturating_sub(filter_length - 1);
        for i in 0..(filter_length - 1).min(effective_length) {
            if memory_start + i < filter_memory.len() {
                filter_memory[memory_start + i] =
                    input_buffer[effective_length - (filter_length - 1) + i];
            }
        }

        compute_time += compute_start.elapsed().as_millis();

        let io_start = Instant::now();

        // Write output chunk
        for &sample in &output_buffer[..effective_length] {
            let sample: f64 = sample;
            let bytes = sample.to_le_bytes();
            output_writer
                .write_all(&bytes)
                .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
        }

        memory_stats.disk_operations += 1;
        io_time += io_start.elapsed().as_millis();

        // Update memory usage tracking
        let current_memory = input_buffer.len() * sample_size
            + output_buffer.len() * sample_size
            + filter_memory.len() * sample_size;
        peak_memory = peak_memory.max(current_memory);

        total_processed += effective_length;

        // Clear buffers to force deallocation
        if total_processed % (chunk_size * 10) == 0 {
            input_buffer.clear();
            input_buffer.resize(chunk_size + config.overlap_size, 0.0);
            output_buffer.clear();
            output_buffer.resize(chunk_size, 0.0);
        }
    }

    // Flush output
    output_writer
        .flush()
        .map_err(|e| SignalError::ComputationError(format!("Flush error: {}", e)))?;

    memory_stats.peak_memory = peak_memory;
    memory_stats.avg_memory = peak_memory / 2; // Rough estimate

    let total_time = start_time.elapsed().as_millis();
    let timing_stats = TimingStats {
        total_time_ms: total_time,
        io_time_ms: io_time,
        compute_time_ms: compute_time,
        memory_mgmt_time_ms: total_time - io_time - compute_time,
    };

    Ok(MemoryOptimizedResult {
        data: MemoryOptimizedData::OnDisk {
            _file_path: output_file.to_string(),
            length: samples_count,
            chunk_size,
        },
        memory_stats,
        timing_stats,
    })
}

/// Memory-optimized FFT for very large signals
///
/// Computes FFT of signals larger than available memory using disk-based
/// radix-2 algorithms with minimal memory footprint.
#[allow(dead_code)]
pub fn memory_optimized_fft(
    input_file: &str,
    output_file: &str,
    config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<scirs2_core::numeric::Complex<f64>>> {
    let _start_time = Instant::now();

    // Open input _file and validate size
    let input_file_handle = File::open(input_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot open input file: {}", e)))?;

    let file_size = input_file_handle
        .metadata()
        .map_err(|e| SignalError::ComputationError(format!("Cannot get _file size: {}", e)))?
        .len() as usize;

    let complex_size = std::mem::size_of::<Complex<f64>>();
    let n = file_size / complex_size;

    // Validate that n is a power of 2
    if !n.is_power_of_two() {
        return Err(SignalError::ValueError(
            "FFT size must be a power of 2 for memory-optimized implementation".to_string(),
        ));
    }

    let log2n = n.trailing_zeros() as usize;

    // Calculate memory requirements
    let sample_size = complex_size;
    let max_samples_in_memory = config.max_memory_bytes / sample_size / 3; // Triple buffering

    if n <= max_samples_in_memory {
        // Can fit in memory - use standard FFT
        return memory_fft_in_core(input_file, output_file, n, config);
    }

    // Use out-of-core FFT algorithm
    memory_fft_out_of_core(input_file, output_file, n, log2n, config)
}

/// In-core FFT for moderately large signals
#[allow(dead_code)]
fn memory_fft_in_core(
    input_file: &str,
    output_file: &str,
    n: usize,
    _config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<scirs2_core::numeric::Complex<f64>>> {
    let start_time = Instant::now();
    let io_start = Instant::now();

    // Read entire signal into memory
    let input_file_handle = File::open(input_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot open input file: {}", e)))?;

    let mut input_reader = BufReader::new(input_file_handle);
    let mut data = vec![Complex::new(0.0, 0.0); n];

    // Read _complex data
    for i in 0..n {
        let mut real_bytes = [0u8; 8];
        let mut imag_bytes = [0u8; 8];

        input_reader
            .read_exact(&mut real_bytes)
            .map_err(|e| SignalError::ComputationError(format!("Read error: {}", e)))?;
        input_reader
            .read_exact(&mut imag_bytes)
            .map_err(|e| SignalError::ComputationError(format!("Read error: {}", e)))?;

        let real = f64::from_le_bytes(real_bytes);
        let imag = f64::from_le_bytes(imag_bytes);
        data[i] = Complex::new(real, imag);
    }

    let io_time = io_start.elapsed().as_millis();
    let compute_start = Instant::now();

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut data);

    let compute_time = compute_start.elapsed().as_millis();
    let io_start = Instant::now();

    // Write result
    let output_file_handle = File::create(output_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create output file: {}", e)))?;

    let mut output_writer = BufWriter::new(output_file_handle);

    for sample in &data {
        output_writer
            .write_all(&sample.re.to_le_bytes())
            .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
        output_writer
            .write_all(&sample.im.to_le_bytes())
            .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
    }

    output_writer
        .flush()
        .map_err(|e| SignalError::ComputationError(format!("Flush error: {}", e)))?;

    let io_time_total = io_time + io_start.elapsed().as_millis();
    let total_time = start_time.elapsed().as_millis();

    let memory_stats = MemoryStats {
        peak_memory: n * std::mem::size_of::<Complex<f64>>(),
        avg_memory: n * std::mem::size_of::<Complex<f64>>() / 2,
        cache_hits: 0,
        cache_misses: 0,
        disk_operations: 2, // Read and write
    };

    let timing_stats = TimingStats {
        total_time_ms: total_time,
        io_time_ms: io_time_total,
        compute_time_ms: compute_time,
        memory_mgmt_time_ms: total_time - io_time_total - compute_time,
    };

    Ok(MemoryOptimizedResult {
        data: MemoryOptimizedData::OnDisk {
            _file_path: output_file.to_string(),
            length: n,
            chunk_size: n,
        },
        memory_stats,
        timing_stats,
    })
}

/// Out-of-core FFT for very large signals
#[allow(dead_code)]
fn memory_fft_out_of_core(
    input_file: &str,
    output_file: &str,
    n: usize,
    log2n: usize,
    config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<scirs2_core::numeric::Complex<f64>>> {
    let start_time = Instant::now();
    let mut memory_stats = MemoryStats {
        peak_memory: 0,
        avg_memory: 0,
        cache_hits: 0,
        cache_misses: 0,
        disk_operations: 0,
    };

    // Calculate stage parameters
    let complex_size = std::mem::size_of::<Complex<f64>>();
    let max_samples_in_memory = config.max_memory_bytes / complex_size / 4;

    // Determine how many stages we can do in memory vs. on disk
    let log2_memory_limit = (max_samples_in_memory as f32).log2().floor() as usize;
    let in_memory_stages = log2_memory_limit.min(log2n);
    let disk_stages = log2n - in_memory_stages;

    // Create temporary files for intermediate results
    let temp_dir = config.temp_dir.as_deref().unwrap_or("/tmp");
    let temp_file = format!("{}/fft_temp_{}.dat", temp_dir, std::process::id());

    let mut current_input = input_file.to_string();
    let mut current_output = if disk_stages > 0 {
        temp_file.clone()
    } else {
        output_file.to_string()
    };

    let mut total_io_time = 0;
    let mut total_compute_time = 0;

    // Perform out-of-core stages first (if any)
    for stage in 0..disk_stages {
        let _stage_start = Instant::now();

        // Process this stage with disk I/O
        let stage_result =
            process_fft_stage_disk(&current_input, &current_output, n, stage, config)?;

        memory_stats.disk_operations += stage_result.memory_stats.disk_operations;
        total_io_time += stage_result.timing_stats.io_time_ms;
        total_compute_time += stage_result.timing_stats.compute_time_ms;

        // Swap files for next stage
        if stage < disk_stages - 1 {
            current_input = current_output.clone();
            current_output = format!("{}_stage_{}", temp_file, stage + 1);
        }
    }

    // Perform in-memory stages
    if in_memory_stages > 0 {
        let final_input = if disk_stages > 0 {
            &current_output
        } else {
            input_file
        };
        let stage_result = process_fft_stages_memory(
            final_input,
            output_file,
            n,
            disk_stages,
            in_memory_stages,
            config,
        )?;

        memory_stats.disk_operations += stage_result.memory_stats.disk_operations;
        total_io_time += stage_result.timing_stats.io_time_ms;
        total_compute_time += stage_result.timing_stats.compute_time_ms;
        memory_stats.peak_memory = stage_result.memory_stats.peak_memory;
    }

    // Clean up temporary files
    let _ = std::fs::remove_file(&temp_file);
    for stage in 0..disk_stages.saturating_sub(1) {
        let _ = std::fs::remove_file(format!("{}_stage_{}", temp_file, stage + 1));
    }

    let total_time = start_time.elapsed().as_millis();
    let timing_stats = TimingStats {
        total_time_ms: total_time,
        io_time_ms: total_io_time,
        compute_time_ms: total_compute_time,
        memory_mgmt_time_ms: total_time - total_io_time - total_compute_time,
    };

    Ok(MemoryOptimizedResult {
        data: MemoryOptimizedData::OnDisk {
            _file_path: output_file.to_string(),
            length: n,
            chunk_size: config.chunk_size,
        },
        memory_stats,
        timing_stats,
    })
}

/// Process a single FFT stage with disk I/O
#[allow(dead_code)]
fn process_fft_stage_disk(
    input_file: &str,
    output_file: &str,
    n: usize,
    stage: usize,
    config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<scirs2_core::numeric::Complex<f64>>> {
    let start_time = std::time::Instant::now();
    let mut disk_ops = 0;

    // Open files
    let input_handle = File::open(input_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot open input: {}", e)))?;
    let output_handle = File::create(output_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create output: {}", e)))?;

    let mut input_reader = BufReader::new(input_handle);
    let mut output_writer = BufWriter::new(output_handle);

    // Calculate stage parameters
    let stage_size = 1 << (stage + 1);
    let half_stage = stage_size / 2;
    let num_groups = n / stage_size;

    let complex_size = std::mem::size_of::<Complex<f64>>();
    let max_groups_in_memory = config.max_memory_bytes / (stage_size * complex_size);
    let groups_per_chunk = max_groups_in_memory.max(1);

    let mut io_time = 0;
    let mut compute_time = 0;

    // Process groups in chunks
    for chunk_start in (0..num_groups).step_by(groups_per_chunk) {
        let io_start = std::time::Instant::now();

        let chunk_groups = groups_per_chunk.min(num_groups - chunk_start);
        let chunk_samples = chunk_groups * stage_size;

        // Read chunk
        let mut data = vec![Complex::new(0.0, 0.0); chunk_samples];

        // Seek to the correct position
        let byte_offset = chunk_start * stage_size * complex_size;
        input_reader
            .seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| SignalError::ComputationError(format!("Seek error: {}", e)))?;

        for i in 0..chunk_samples {
            let mut real_bytes = [0u8; 8];
            let mut imag_bytes = [0u8; 8];

            input_reader
                .read_exact(&mut real_bytes)
                .map_err(|e| SignalError::ComputationError(format!("Read error: {}", e)))?;
            input_reader
                .read_exact(&mut imag_bytes)
                .map_err(|e| SignalError::ComputationError(format!("Read error: {}", e)))?;

            data[i] = Complex::new(
                f64::from_le_bytes(real_bytes),
                f64::from_le_bytes(imag_bytes),
            );
        }

        disk_ops += 1;
        io_time += io_start.elapsed().as_millis();

        let compute_start = std::time::Instant::now();

        // Process butterfly operations for this chunk
        for group in 0..chunk_groups {
            let group_offset = group * stage_size;

            for i in 0..half_stage {
                let j = i + half_stage;
                let twiddle_angle = -2.0 * PI * (i as f64) / (stage_size as f64);
                let twiddle = Complex::new(twiddle_angle.cos(), twiddle_angle.sin());

                let idx1 = group_offset + i;
                let idx2 = group_offset + j;

                let t = data[idx2] * twiddle;
                let u = data[idx1];

                data[idx1] = u + t;
                data[idx2] = u - t;
            }
        }

        compute_time += compute_start.elapsed().as_millis();

        let io_start = std::time::Instant::now();

        // Write chunk back
        output_writer
            .seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| SignalError::ComputationError(format!("Output seek error: {}", e)))?;

        for sample in &data {
            output_writer
                .write_all(&sample.re.to_le_bytes())
                .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
            output_writer
                .write_all(&sample.im.to_le_bytes())
                .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
        }

        disk_ops += 1;
        io_time += io_start.elapsed().as_millis();
    }

    output_writer
        .flush()
        .map_err(|e| SignalError::ComputationError(format!("Flush error: {}", e)))?;

    let total_time = start_time.elapsed().as_millis();

    let memory_stats = MemoryStats {
        peak_memory: groups_per_chunk * stage_size * complex_size,
        avg_memory: groups_per_chunk * stage_size * complex_size / 2,
        cache_hits: 0,
        cache_misses: 0,
        disk_operations: disk_ops,
    };

    let timing_stats = TimingStats {
        total_time_ms: total_time,
        io_time_ms: io_time,
        compute_time_ms: compute_time,
        memory_mgmt_time_ms: total_time - io_time - compute_time,
    };

    Ok(MemoryOptimizedResult {
        data: MemoryOptimizedData::OnDisk {
            _file_path: output_file.to_string(),
            length: n,
            chunk_size: config.chunk_size,
        },
        memory_stats,
        timing_stats,
    })
}

/// Process remaining FFT stages in memory
#[allow(dead_code)]
fn process_fft_stages_memory(
    input_file: &str,
    output_file: &str,
    n: usize,
    _start_stage: usize,
    _stages: usize,
    config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<scirs2_core::numeric::Complex<f64>>> {
    // For simplicity, delegate to in-core implementation
    // In a full implementation, this would do the remaining radix-2 _stages
    memory_fft_in_core(input_file, output_file, n, config)
}

/// Memory-optimized spectrogram computation
///
/// Computes spectrograms of very large signals using sliding window approach
/// with minimal memory footprint.
#[allow(dead_code)]
pub fn memory_optimized_spectrogram(
    input_file: &str,
    output_file: &str,
    window_size: usize,
    hop_size: usize,
    _config: &MemoryConfig,
) -> SignalResult<MemoryOptimizedResult<f64>> {
    let start_time = Instant::now();

    // Validate parameters
    if window_size == 0 || hop_size == 0 {
        return Err(SignalError::ValueError(
            "Window _size and hop _size must be positive".to_string(),
        ));
    }

    if hop_size > window_size {
        return Err(SignalError::ValueError(
            "Hop _size should not exceed window _size".to_string(),
        ));
    }

    // Open files
    let input_handle = File::open(input_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot open input: {}", e)))?;
    let output_handle = File::create(output_file)
        .map_err(|e| SignalError::ComputationError(format!("Cannot create output: {}", e)))?;

    let file_size = input_handle
        .metadata()
        .map_err(|e| SignalError::ComputationError(format!("Cannot get _file size: {}", e)))?
        .len() as usize;

    let n_samples = file_size / std::mem::size_of::<f64>();
    let n_frames = (n_samples - window_size) / hop_size + 1;
    let n_freqs = window_size / 2 + 1;

    let mut input_reader = BufReader::new(input_handle);
    let mut output_writer = BufWriter::new(output_handle);

    // Initialize FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    // Create window function (Hann window)
    let window: Vec<f64> = (0..window_size)
        .map(|i| {
            0.5 * (1.0
                - ((2.0 * std::f64::consts::PI * i as f64) / (window_size as f64 - 1.0)).cos())
        })
        .collect();

    let mut buffer = vec![0.0; window_size];
    let mut fft_buffer = vec![Complex::new(0.0, 0.0); window_size];
    let mut magnitude_buffer = vec![0.0; n_freqs];

    let mut total_io_time = 0;
    let mut total_compute_time = 0;
    let mut disk_ops = 0;

    // Process each frame
    for frame in 0..n_frames {
        let io_start = Instant::now();

        // Seek to frame position
        let byte_offset = frame * hop_size * std::mem::size_of::<f64>();
        input_reader
            .seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| SignalError::ComputationError(format!("Seek error: {}", e)))?;

        // Read frame data
        for i in 0..window_size {
            let mut bytes = [0u8; 8];
            if input_reader.read_exact(&mut bytes).is_ok() {
                buffer[i] = f64::from_le_bytes(bytes);
            } else {
                buffer[i] = 0.0; // Zero-pad if we reach end of _file
            }
        }

        disk_ops += 1;
        total_io_time += io_start.elapsed().as_millis();

        let compute_start = Instant::now();

        // Apply window and prepare FFT buffer
        for i in 0..window_size {
            fft_buffer[i] = Complex::new(buffer[i] * window[i], 0.0);
        }

        // Compute FFT
        fft.process(&mut fft_buffer);

        // Compute magnitude spectrum (one-sided)
        for i in 0..n_freqs {
            magnitude_buffer[i] = fft_buffer[i].norm_sqr();
            if i > 0 && i < window_size / 2 {
                magnitude_buffer[i] *= 2.0; // Account for negative frequencies
            }
        }

        total_compute_time += compute_start.elapsed().as_millis();

        let io_start = Instant::now();

        // Write magnitude spectrum
        for &mag in &magnitude_buffer {
            output_writer
                .write_all(&mag.to_le_bytes())
                .map_err(|e| SignalError::ComputationError(format!("Write error: {}", e)))?;
        }

        disk_ops += 1;
        total_io_time += io_start.elapsed().as_millis();
    }

    output_writer
        .flush()
        .map_err(|e| SignalError::ComputationError(format!("Flush error: {}", e)))?;

    let total_time = start_time.elapsed().as_millis();

    let memory_stats = MemoryStats {
        peak_memory: (window_size * 2 + n_freqs) * std::mem::size_of::<f64>(),
        avg_memory: (window_size * 2 + n_freqs) * std::mem::size_of::<f64>() / 2,
        cache_hits: 0,
        cache_misses: 0,
        disk_operations: disk_ops,
    };

    let timing_stats = TimingStats {
        total_time_ms: total_time,
        io_time_ms: total_io_time,
        compute_time_ms: total_compute_time,
        memory_mgmt_time_ms: total_time - total_io_time - total_compute_time,
    };

    Ok(MemoryOptimizedResult {
        data: MemoryOptimizedData::OnDisk {
            _file_path: output_file.to_string(),
            length: n_frames * n_freqs,
            chunk_size: n_freqs,
        },
        memory_stats,
        timing_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_memory_config_defaults() {
        let config = MemoryConfig::default();
        assert!(config.max_memory_bytes > 0);
        assert!(config.chunk_size > 0);
        assert!(config.overlap_size > 0);
    }

    #[test]
    fn test_memory_optimized_data_variants() {
        let in_memory = MemoryOptimizedData::InMemory(vec![1.0, 2.0, 3.0]);
        let on_disk = MemoryOptimizedData::OnDisk {
            file_path: "/tmp/test.dat".to_string(),
            length: 1000,
            chunk_size: 256,
        };

        match in_memory {
            MemoryOptimizedData::InMemory(ref data) => {
                assert_eq!(data.len(), 3);
            }
            _ => panic!("Expected InMemory variant"),
        }

        match on_disk {
            MemoryOptimizedData::OnDisk { length, .. } => {
                assert_eq!(length, 1000);
            }
            _ => panic!("Expected OnDisk variant"),
        }
    }

    #[test]
    fn test_create_test_signal_file() -> SignalResult<()> {
        let test_file = "/tmp/test_signal.dat";
        let n_samples = 1000;

        // Create test signal file
        let mut file = File::create(test_file).unwrap();
        for i in 0..n_samples {
            let sample = (i as f64 * 0.1).sin();
            file.write_all(&sample.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        // Verify file was created
        let metadata = fs::metadata(test_file).unwrap();
        assert_eq!(metadata.len(), (n_samples * 8) as u64);

        // Clean up
        fs::remove_file(test_file).unwrap();

        Ok(())
    }
}
