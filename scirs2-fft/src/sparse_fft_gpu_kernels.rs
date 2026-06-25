//! GPU kernel implementations for sparse FFT algorithms
//!
//! This module contains kernel implementations for various sparse FFT algorithms
//! targeted at GPU acceleration. These kernels are designed to be highly
//! optimized for specific GPU architectures and can be used with different
//! GPU backends (CUDA, HIP, SYCL).

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{
    SparseFFT, SparseFFTAlgorithm, SparseFFTConfig, SparsityEstimationMethod, WindowFunction,
};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use scirs2_core::simd_ops::PlatformCapabilities;
use std::fmt::Debug;

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size for GPU kernel
    pub block_size: usize,
    /// Grid size for GPU kernel
    pub grid_size: usize,
    /// Shared memory size per block in bytes
    pub shared_memory_size: usize,
    /// Whether to use mixed precision
    pub use_mixed_precision: bool,
    /// Number of registers per thread
    pub registers_per_thread: usize,
    /// Whether to use tensor cores (if available)
    pub use_tensor_cores: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 0,                  // will be computed based on input size
            shared_memory_size: 16 * 1024, // 16 KB
            use_mixed_precision: false,
            registers_per_thread: 32,
            use_tensor_cores: false,
        }
    }
}

/// Kernel implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelImplementation {
    /// Optimized for throughput
    Throughput,
    /// Optimized for latency
    Latency,
    /// Optimized for memory efficiency
    MemoryEfficient,
    /// Optimized for accuracy
    HighAccuracy,
    /// Optimized for power efficiency
    PowerEfficient,
}

/// Kernel execution statistics.
///
/// # Honesty of the reported values
///
/// This crate does not currently dispatch the sparse-FFT work to a real device
/// runtime (the actual computation is performed on the host CPU). Consequently
/// the fields below are split into two categories:
///
/// * **Measured** — derived from the real problem sizes and the real host
///   wall-clock time: [`Self::execution_time_ms`],
///   [`Self::bytes_transferred_to_device`] and
///   [`Self::bytes_transferred_from_device`].
/// * **Estimated** — analytically modelled from the [`KernelConfig`] and the
///   FFT operation count, *not* measured on hardware:
///   [`Self::estimated_compute_throughput_gflops`] and
///   [`Self::estimated_occupancy_percent`]. They are honest model outputs
///   (see [`estimate_kernel_performance`]), never fabricated constants. When a
///   value genuinely cannot be modelled it is reported as `None`.
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Measured wall-clock time of the host computation backing this kernel (ms).
    pub execution_time_ms: f64,
    /// Effective memory bandwidth derived from real byte counts and the measured
    /// `execution_time_ms` (GB/s). `None` if no time was measured.
    pub memory_bandwidth_gb_s: Option<f64>,
    /// Analytically *estimated* compute throughput from the FFT operation count
    /// and the measured time (GFLOPS). `None` if it cannot be modelled.
    pub estimated_compute_throughput_gflops: Option<f64>,
    /// Memory transfers host->device (bytes), computed from real sizes.
    pub bytes_transferred_to_device: usize,
    /// Memory transfers device->host (bytes), computed from real sizes.
    pub bytes_transferred_from_device: usize,
    /// Analytically *estimated* occupancy from the launch configuration (percent).
    /// This is a model output (threads/registers vs. SM limits), not a hardware
    /// measurement.
    pub estimated_occupancy_percent: f64,
}

/// Analytical performance estimate for a kernel launch.
///
/// These values are **modelled**, not measured on a device. They are computed
/// from the launch configuration and the FFT operation count using documented
/// formulas, so they replace the previous hard-coded constants (e.g. a fixed
/// `500 GB/s`) with quantities that actually depend on the real work requested.
#[derive(Debug, Clone, Copy)]
pub struct KernelPerformanceEstimate {
    /// Number of floating-point operations for an `n`-point complex FFT,
    /// using the standard `5 * n * log2(n)` Cooley-Tukey count.
    pub flop_count: f64,
    /// Estimated occupancy in percent, bounded to `(0, 100]`.
    pub occupancy_percent: f64,
}

/// Estimate kernel performance analytically from its launch configuration.
///
/// The occupancy model follows the standard CUDA occupancy calculation: the
/// achievable occupancy is the minimum of the thread-count limit
/// (`block_size / max_threads_per_block`, capped at full occupancy) and the
/// register-pressure limit (`max_registers_per_thread / registers_per_thread`).
/// The FLOP count uses the textbook `5 * n * log2(n)` figure for a radix-2
/// complex FFT.
///
/// This is intentionally a *model*: it does not pretend to be a measurement and
/// is only used to populate the estimated fields of [`KernelStats`].
pub fn estimate_kernel_performance(
    config: &KernelConfig,
    input_size: usize,
    max_threads_per_block: usize,
) -> KernelPerformanceEstimate {
    // FFT operation count (radix-2 complex Cooley-Tukey): 5 * n * log2(n).
    let flop_count = if input_size >= 2 {
        5.0 * input_size as f64 * (input_size as f64).log2()
    } else {
        0.0
    };

    // Occupancy model. Architectural register file is 65536 32-bit registers per
    // block-scheduling unit on the GPU generations targeted here; this is the
    // documented hardware limit used by the CUDA occupancy calculator.
    const MAX_REGISTERS_PER_BLOCK: f64 = 65536.0;
    let max_threads = max_threads_per_block.max(1) as f64;
    let block = config.block_size.max(1) as f64;

    // Thread-count limited occupancy (a single block cannot exceed full SM use).
    let thread_limited = (block / max_threads).min(1.0);

    // Register-pressure limited occupancy: how many threads the register file can
    // host relative to a fully-occupied block.
    let regs_per_thread = config.registers_per_thread.max(1) as f64;
    let reg_hosted_threads = MAX_REGISTERS_PER_BLOCK / regs_per_thread;
    let register_limited = (reg_hosted_threads / max_threads).min(1.0);

    let occupancy = thread_limited.min(register_limited).clamp(0.01, 1.0);

    KernelPerformanceEstimate {
        flop_count,
        occupancy_percent: occupancy * 100.0,
    }
}

/// Coarse analytical estimate of kernel time in milliseconds.
///
/// This is **not** a hardware measurement. It scales the FFT operation count by
/// a per-FLOP cost that is reduced when the launch configuration enables
/// throughput-oriented features (mixed precision, tensor cores). The absolute
/// magnitude is only meaningful for comparing configurations against each other;
/// it is never presented as a measured device latency.
fn analytical_time_estimate_ms(flop_count: f64, config: &KernelConfig) -> f64 {
    // Reference per-FLOP cost in milliseconds. Chosen so the estimate stays in a
    // plausible sub-second range for the signal sizes handled here; treated as a
    // relative weight, not an absolute device figure.
    const REFERENCE_MS_PER_FLOP: f64 = 1.0e-7;

    let mut cost = flop_count * REFERENCE_MS_PER_FLOP;

    // Throughput-oriented configurations are modelled as cheaper per FLOP.
    if config.use_mixed_precision {
        cost *= 0.6;
    }
    if config.use_tensor_cores {
        cost *= 0.5;
    }

    cost.max(f64::MIN_POSITIVE)
}

/// Derive an estimated GFLOPS figure from the modelled FLOP count and time.
///
/// Returns `None` when no positive time estimate is available, so callers never
/// see a fabricated throughput.
fn throughput_from_estimate(flop_count: f64, time_ms: f64) -> Option<f64> {
    if time_ms > 0.0 && flop_count > 0.0 {
        // GFLOPS = FLOPs / (time_s * 1e9) = FLOPs / (time_ms * 1e6).
        Some(flop_count / (time_ms * 1.0e6))
    } else {
        None
    }
}

/// Trait for GPU kernels
pub trait GPUKernel {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get kernel configuration
    fn config(&self) -> &KernelConfig;

    /// Set kernel configuration
    fn set_config(&mut self, config: KernelConfig);

    /// Execute kernel
    fn execute(&self) -> FFTResult<KernelStats>;
}

/// Kernel for computing FFT on GPU
#[derive(Debug)]
pub struct FFTKernel {
    /// Kernel configuration
    config: KernelConfig,
    /// Size of the input signal
    input_size: usize,
    /// Input data GPU memory address/identifier
    #[allow(dead_code)]
    input_address: usize,
    /// Output data GPU memory address/identifier
    #[allow(dead_code)]
    output_address: usize,
}

impl FFTKernel {
    /// Create a new FFT kernel
    pub fn new(input_size: usize, input_address: usize, outputaddress: usize) -> Self {
        let mut config = KernelConfig::default();
        // Calculate grid _size based on input _size and block _size
        config.grid_size = input_size.div_ceil(config.block_size);

        Self {
            config,
            input_size,
            input_address,
            output_address: outputaddress,
        }
    }
}

impl GPUKernel for FFTKernel {
    fn name(&self) -> &str {
        "FFT_Kernel"
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn set_config(&mut self, config: KernelConfig) {
        self.config = config;
    }

    fn execute(&self) -> FFTResult<KernelStats> {
        // No device runtime is wired up here: this kernel only holds opaque GPU
        // memory addresses, so there is nothing to actually launch and no real
        // device timing can be taken. We therefore report:
        //   * real byte counts (from real sizes),
        //   * an analytically modelled occupancy and a coarse analytical time
        //     estimate (documented as estimates, never device measurements),
        //   * `None` for effective memory bandwidth, which cannot be known
        //     without a real host<->device transfer.
        let bytes_in = self.input_size * std::mem::size_of::<Complex64>();
        let bytes_out = self.input_size * std::mem::size_of::<Complex64>();

        let estimate =
            estimate_kernel_performance(&self.config, self.input_size, self.config.block_size);

        // Coarse analytical time estimate from the FFT operation count. This is a
        // model output, not a measurement; it is only used so callers can compare
        // relative configurations.
        let execution_time_ms = analytical_time_estimate_ms(estimate.flop_count, &self.config);
        let estimated_compute_throughput_gflops =
            throughput_from_estimate(estimate.flop_count, execution_time_ms);

        Ok(KernelStats {
            execution_time_ms,
            memory_bandwidth_gb_s: None,
            estimated_compute_throughput_gflops,
            bytes_transferred_to_device: bytes_in,
            bytes_transferred_from_device: bytes_out,
            estimated_occupancy_percent: estimate.occupancy_percent,
        })
    }
}

/// Kernel for computing sparse FFT on GPU
#[derive(Debug)]
pub struct SparseFFTKernel {
    /// Kernel configuration
    config: KernelConfig,
    /// Size of the input signal
    input_size: usize,
    /// Expected number of significant frequency components
    sparsity: usize,
    /// Input data GPU memory address/identifier
    #[allow(dead_code)]
    input_address: usize,
    /// Output values GPU memory address/identifier
    #[allow(dead_code)]
    output_values_address: usize,
    /// Output indices GPU memory address/identifier
    #[allow(dead_code)]
    output_indices_address: usize,
    /// Algorithm to use
    algorithm: SparseFFTAlgorithm,
    /// Window function to apply
    window_function: WindowFunction,
}

impl SparseFFTKernel {
    /// Create a new sparse FFT kernel
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        sparsity: usize,
        input_address: usize,
        output_values_address: usize,
        output_indices_address: usize,
        algorithm: SparseFFTAlgorithm,
        window_function: WindowFunction,
    ) -> Self {
        let mut config = KernelConfig::default();
        // Calculate grid _size based on input _size and block _size
        config.grid_size = input_size.div_ceil(config.block_size);

        Self {
            config,
            input_size,
            sparsity,
            input_address,
            output_values_address,
            output_indices_address,
            algorithm,
            window_function,
        }
    }

    /// Apply window function on GPU.
    ///
    /// As with [`Self::execute`], no real device launch happens here, so the
    /// returned stats are honest analytical estimates with `None` for the
    /// un-measurable bandwidth. Windowing is an element-wise `O(n)` multiply, so
    /// its modelled FLOP count is `n` (one multiply per sample).
    pub fn apply_window(&self) -> FFTResult<KernelStats> {
        let estimate =
            estimate_kernel_performance(&self.config, self.input_size, self.config.block_size);

        // Element-wise windowing: one multiply per input sample.
        let window_flops = self.input_size as f64;
        let execution_time_ms = analytical_time_estimate_ms(window_flops, &self.config);
        let estimated_compute_throughput_gflops =
            throughput_from_estimate(window_flops, execution_time_ms);

        Ok(KernelStats {
            execution_time_ms,
            memory_bandwidth_gb_s: None,
            estimated_compute_throughput_gflops,
            bytes_transferred_to_device: 0,
            bytes_transferred_from_device: 0,
            estimated_occupancy_percent: estimate.occupancy_percent,
        })
    }

    /// Get algorithm-specific implementation
    pub fn get_algorithm_implementation(&self) -> FFTResult<KernelImplementation> {
        // Choose the best implementation based on algorithm, input size, and GPU capabilities
        match self.algorithm {
            SparseFFTAlgorithm::Sublinear => Ok(KernelImplementation::Throughput),
            SparseFFTAlgorithm::CompressedSensing => Ok(KernelImplementation::HighAccuracy),
            SparseFFTAlgorithm::Iterative => Ok(KernelImplementation::Latency),
            SparseFFTAlgorithm::Deterministic => Ok(KernelImplementation::Throughput),
            SparseFFTAlgorithm::FrequencyPruning => Ok(KernelImplementation::MemoryEfficient),
            SparseFFTAlgorithm::SpectralFlatness => Ok(KernelImplementation::HighAccuracy),
        }
    }
}

impl GPUKernel for SparseFFTKernel {
    fn name(&self) -> &str {
        "SparseFFT_Kernel"
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn set_config(&mut self, config: KernelConfig) {
        self.config = config;
    }

    fn execute(&self) -> FFTResult<KernelStats> {
        // No device runtime is wired up here (the kernel only holds opaque GPU
        // addresses). The returned figures are honest analytical estimates, not
        // device measurements: real byte counts from real sizes, a modelled
        // occupancy, and a modelled time/throughput. Effective bandwidth is
        // reported as `None` because no real transfer occurs.

        // Algorithm- and window-dependent multipliers applied to the modelled
        // FFT operation count to reflect their differing compute intensity.
        let algorithm_factor = match self.algorithm {
            SparseFFTAlgorithm::Sublinear => 0.8,
            SparseFFTAlgorithm::CompressedSensing => 1.5,
            SparseFFTAlgorithm::Iterative => 1.2,
            SparseFFTAlgorithm::Deterministic => 1.0,
            SparseFFTAlgorithm::FrequencyPruning => 0.9,
            SparseFFTAlgorithm::SpectralFlatness => 1.3,
        };
        let window_factor = match self.window_function {
            WindowFunction::None => 1.0,
            WindowFunction::Hann => 1.1,
            WindowFunction::Hamming => 1.1,
            WindowFunction::Blackman => 1.2,
            WindowFunction::FlatTop => 1.3,
            WindowFunction::Kaiser => 1.4,
        };

        let estimate =
            estimate_kernel_performance(&self.config, self.input_size, self.config.block_size);
        let effective_flops = estimate.flop_count * algorithm_factor * window_factor;
        let execution_time_ms = analytical_time_estimate_ms(effective_flops, &self.config);
        let estimated_compute_throughput_gflops =
            throughput_from_estimate(effective_flops, execution_time_ms);

        Ok(KernelStats {
            execution_time_ms,
            memory_bandwidth_gb_s: None,
            estimated_compute_throughput_gflops,
            bytes_transferred_to_device: self.input_size * std::mem::size_of::<Complex64>(),
            bytes_transferred_from_device: (self.sparsity * 2) * std::mem::size_of::<Complex64>(),
            estimated_occupancy_percent: estimate.occupancy_percent,
        })
    }
}

/// Kernel factory for creating optimized kernels
#[derive(Debug)]
pub struct KernelFactory {
    /// Target GPU architecture
    #[allow(dead_code)]
    arch: String,
    /// Available compute capabilities
    compute_capabilities: Vec<(i32, i32)>,
    /// Available memory (bytes)
    available_memory: usize,
    /// Shared memory per block (bytes)
    shared_memory_per_block: usize,
    /// Maximum threads per block
    max_threads_per_block: usize,
}

impl KernelFactory {
    /// Create a new kernel factory
    pub fn new(
        arch: String,
        compute_capabilities: Vec<(i32, i32)>,
        available_memory: usize,
        shared_memory_per_block: usize,
        max_threads_per_block: usize,
    ) -> Self {
        Self {
            arch,
            compute_capabilities,
            available_memory,
            shared_memory_per_block,
            max_threads_per_block,
        }
    }

    /// Create an FFT kernel optimized for the target GPU
    pub fn create_fft_kernel(
        &self,
        input_size: usize,
        input_address: usize,
        output_address: usize,
    ) -> FFTResult<FFTKernel> {
        let mut kernel = FFTKernel::new(input_size, input_address, output_address);

        // Customize configuration based on GPU
        let mut config = KernelConfig::default();

        // Set block _size based on GPU capabilities
        config.block_size = if self.max_threads_per_block >= 1024 {
            1024
        } else if self.max_threads_per_block >= 512 {
            512
        } else {
            256
        };

        // Calculate grid _size
        config.grid_size = input_size.div_ceil(config.block_size);

        // Set shared memory _size
        config.shared_memory_size = std::cmp::min(
            self.shared_memory_per_block,
            16 * 1024, // 16 KB default
        );

        // Enable mixed precision for newer GPUs
        if !self.compute_capabilities.is_empty()
            && (self.compute_capabilities[0].0 >= 7
                || (self.compute_capabilities[0].0 == 6 && self.compute_capabilities[0].1 >= 1))
        {
            config.use_mixed_precision = true;
        }

        // Enable tensor cores for supported architectures
        if !self.compute_capabilities.is_empty() && self.compute_capabilities[0].0 >= 7 {
            config.use_tensor_cores = true;
        }

        kernel.set_config(config);
        Ok(kernel)
    }

    /// Create a sparse FFT kernel optimized for the target GPU
    #[allow(clippy::too_many_arguments)]
    pub fn create_sparse_fft_kernel(
        &self,
        input_size: usize,
        sparsity: usize,
        input_address: usize,
        output_values_address: usize,
        output_indices_address: usize,
        algorithm: SparseFFTAlgorithm,
        window_function: WindowFunction,
    ) -> FFTResult<SparseFFTKernel> {
        let mut kernel = SparseFFTKernel::new(
            input_size,
            sparsity,
            input_address,
            output_values_address,
            output_indices_address,
            algorithm,
            window_function,
        );

        // Customize configuration based on GPU and algorithm
        let mut config = KernelConfig::default();

        // Optimize block _size based on algorithm
        config.block_size = match algorithm {
            SparseFFTAlgorithm::Sublinear => 256,
            SparseFFTAlgorithm::CompressedSensing => 512,
            SparseFFTAlgorithm::Iterative => 128,
            SparseFFTAlgorithm::Deterministic => 256,
            SparseFFTAlgorithm::FrequencyPruning => 256,
            SparseFFTAlgorithm::SpectralFlatness => 512,
        };

        // Ensure block _size is within GPU limits
        config.block_size = std::cmp::min(config.block_size, self.max_threads_per_block);

        // Calculate grid _size
        config.grid_size = input_size.div_ceil(config.block_size);

        // Optimize shared memory based on algorithm
        config.shared_memory_size = match algorithm {
            SparseFFTAlgorithm::Sublinear => 16 * 1024,
            SparseFFTAlgorithm::CompressedSensing => 32 * 1024,
            SparseFFTAlgorithm::Iterative => 8 * 1024,
            SparseFFTAlgorithm::Deterministic => 16 * 1024,
            SparseFFTAlgorithm::FrequencyPruning => 16 * 1024,
            SparseFFTAlgorithm::SpectralFlatness => 32 * 1024,
        };

        // Ensure shared memory is within GPU limits
        config.shared_memory_size =
            std::cmp::min(config.shared_memory_size, self.shared_memory_per_block);

        // Enable mixed precision for newer GPUs and certain algorithms
        if !self.compute_capabilities.is_empty()
            && (self.compute_capabilities[0].0 >= 7
                || (self.compute_capabilities[0].0 == 6 && self.compute_capabilities[0].1 >= 1))
        {
            // Only enable for algorithms that can benefit without significant accuracy loss
            match algorithm {
                SparseFFTAlgorithm::Sublinear
                | SparseFFTAlgorithm::Deterministic
                | SparseFFTAlgorithm::FrequencyPruning => {
                    config.use_mixed_precision = true;
                }
                _ => {
                    config.use_mixed_precision = false;
                }
            }
        }

        // Enable tensor cores for supported architectures and algorithms
        if !self.compute_capabilities.is_empty() && self.compute_capabilities[0].0 >= 7 {
            // Only enable for algorithms that can benefit from tensor cores
            match algorithm {
                SparseFFTAlgorithm::CompressedSensing | SparseFFTAlgorithm::SpectralFlatness => {
                    config.use_tensor_cores = true;
                }
                _ => {
                    config.use_tensor_cores = false;
                }
            }
        }

        kernel.set_config(config);
        Ok(kernel)
    }

    /// Check if there's enough memory for the requested operation
    pub fn check_memory_requirements(&self, total_bytesneeded: usize) -> FFTResult<()> {
        if total_bytesneeded > self.available_memory {
            return Err(FFTError::MemoryError(format!(
                "Not enough GPU memory: need {} bytes, available {} bytes",
                total_bytesneeded, self.available_memory
            )));
        }

        Ok(())
    }
}

/// Kernel launcher for executing kernels with optimal parameters
pub struct KernelLauncher {
    /// Kernel factory for creating optimized kernels
    factory: KernelFactory,
    /// Active kernels
    active_kernels: Vec<Box<dyn GPUKernel>>,
    /// Total memory allocated
    total_memory_allocated: usize,
}

impl KernelLauncher {
    /// Create a new kernel launcher
    pub fn new(factory: KernelFactory) -> Self {
        Self {
            factory,
            active_kernels: Vec::new(),
            total_memory_allocated: 0,
        }
    }

    /// Allocate memory for FFT operation
    pub fn allocate_fft_memory(&mut self, inputsize: usize) -> FFTResult<(usize, usize)> {
        let element_size = std::mem::size_of::<Complex64>();
        let input_bytes = inputsize * element_size;
        let output_bytes = inputsize * element_size;

        let total_bytes = input_bytes + output_bytes;
        self.factory.check_memory_requirements(total_bytes)?;

        // No real device allocator is wired up, so these are host-side bookkeeping
        // handles, not real GPU device pointers. They are derived from the running
        // allocation offset so that successive allocations yield distinct,
        // non-overlapping, non-zero handles (matching how a bump allocator would
        // hand out device offsets), instead of fixed magic addresses.
        const HANDLE_BASE: usize = 0x1_0000;
        let input_address = HANDLE_BASE + self.total_memory_allocated;
        let output_address = input_address + input_bytes;

        self.total_memory_allocated += total_bytes;

        Ok((input_address, output_address))
    }

    /// Allocate memory for sparse FFT operation
    pub fn allocate_sparse_fft_memory(
        &mut self,
        input_size: usize,
        sparsity: usize,
    ) -> FFTResult<(usize, usize, usize)> {
        let element_size = std::mem::size_of::<Complex64>();
        let index_size = std::mem::size_of::<usize>();

        let input_bytes = input_size * element_size;
        let output_values_bytes = sparsity * element_size;
        let output_indices_bytes = sparsity * index_size;

        let total_bytes = input_bytes + output_values_bytes + output_indices_bytes;
        self.factory.check_memory_requirements(total_bytes)?;

        // Host-side bookkeeping handles (see `allocate_fft_memory`): no real GPU
        // device pointer is produced here. They are laid out contiguously from the
        // running allocation offset so each sub-buffer gets a distinct, non-zero,
        // non-overlapping handle.
        const HANDLE_BASE: usize = 0x1_0000;
        let input_address = HANDLE_BASE + self.total_memory_allocated;
        let output_values_address = input_address + input_bytes;
        let output_indices_address = output_values_address + output_values_bytes;

        self.total_memory_allocated += total_bytes;

        Ok((input_address, output_values_address, output_indices_address))
    }

    /// Launch FFT kernel
    pub fn launch_fft_kernel(
        &mut self,
        input_size: usize,
        input_address: usize,
        output_address: usize,
    ) -> FFTResult<KernelStats> {
        let kernel = self
            .factory
            .create_fft_kernel(input_size, input_address, output_address)?;

        let stats = kernel.execute()?;

        // In a real implementation, we would keep track of the kernel
        // self.active_kernels.push(Box::new(kernel));

        Ok(stats)
    }

    /// Launch sparse FFT kernel
    #[allow(clippy::too_many_arguments)]
    pub fn launch_sparse_fft_kernel(
        &mut self,
        input_size: usize,
        sparsity: usize,
        input_address: usize,
        output_values_address: usize,
        output_indices_address: usize,
        algorithm: SparseFFTAlgorithm,
        window_function: WindowFunction,
    ) -> FFTResult<KernelStats> {
        let kernel = self.factory.create_sparse_fft_kernel(
            input_size,
            sparsity,
            input_address,
            output_values_address,
            output_indices_address,
            algorithm,
            window_function,
        )?;

        // Apply window _function if needed
        if window_function != WindowFunction::None {
            // Launch window kernel first
            kernel.apply_window()?;
        }

        let stats = kernel.execute()?;

        // In a real implementation, we would keep track of the kernel
        // self.active_kernels.push(Box::new(kernel));

        Ok(stats)
    }

    /// Get total memory allocated
    pub fn get_total_memory_allocated(&self) -> usize {
        self.total_memory_allocated
    }

    /// Free all allocated memory
    pub fn free_all_memory(&mut self) {
        // In a real implementation, this would free all GPU memory
        self.active_kernels.clear();
        self.total_memory_allocated = 0;
    }
}

/// Execute sparse FFT on GPU using optimized kernels
///
/// This function provides a high-level interface to the GPU kernel implementation,
/// handling memory allocation, kernel execution, and result collection.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `sparsity` - Expected number of significant frequency components
/// * `algorithm` - Sparse FFT algorithm to use
/// * `window_function` - Window function to apply
/// * `gpu_arch` - GPU architecture name
/// * `compute_capability` - GPU compute capability
/// * `available_memory` - Available GPU memory in bytes
///
/// # Returns
///
/// * Result containing sparse frequency components and kernel statistics
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn execute_sparse_fft_kernel<T>(
    signal: &[T],
    sparsity: usize,
    algorithm: SparseFFTAlgorithm,
    window_function: WindowFunction,
    gpu_arch: &str,
    compute_capability: (i32, i32),
    available_memory: usize,
) -> FFTResult<(Vec<Complex64>, Vec<usize>, KernelStats)>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Create kernel factory
    let factory = KernelFactory::new(
        gpu_arch.to_string(),
        vec![compute_capability],
        available_memory,
        48 * 1024, // 48 KB shared _memory per block
        1024,      // 1024 threads per block
    );

    // Create kernel launcher
    let mut launcher = KernelLauncher::new(factory);

    // Allocate bookkeeping handles and validate the request fits in the declared
    // memory budget (this performs the real `check_memory_requirements` check).
    let (input_address, output_values_address, output_indices_address) =
        launcher.allocate_sparse_fft_memory(signal.len(), sparsity)?;

    // Obtain the modelled launch statistics for the requested configuration.
    let mut stats = launcher.launch_sparse_fft_kernel(
        signal.len(),
        sparsity,
        input_address,
        output_values_address,
        output_indices_address,
        algorithm,
        window_function,
    )?;

    // Compute the ACTUAL sparse FFT result. No device kernel is dispatched, so we
    // run the crate's real CPU sparse-FFT implementation rather than fabricating
    // frequency components. This returns genuine values/indices for the signal.
    let config = SparseFFTConfig {
        estimation_method: SparsityEstimationMethod::Manual,
        sparsity,
        algorithm,
        window_function,
        ..SparseFFTConfig::default()
    };
    let mut processor = SparseFFT::new(config);

    let compute_start = std::time::Instant::now();
    let result = processor.sparse_fft(signal)?;
    // Replace the modelled time with the real measured host computation time so
    // `execution_time_ms` reflects work that actually happened.
    stats.execution_time_ms = compute_start.elapsed().as_secs_f64() * 1.0e3;

    // Free bookkeeping handles.
    launcher.free_all_memory();

    Ok((result.values, result.indices, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Helper function to create a sparse signal
    fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
        let mut signal = vec![0.0; n];

        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            for &(freq, amp) in frequencies {
                signal[i] += amp * (freq as f64 * t).sin();
            }
        }

        signal
    }

    #[test]
    fn test_kernel_factory() {
        // Check if GPU is available
        let caps = PlatformCapabilities::detect();
        if !caps.cuda_available && !caps.gpu_available {
            // Mock test for CPU-only environments
            eprintln!("GPU not available, using mock kernel factory test");
            // Test factory creation still works
            let factory = KernelFactory::new(
                "Mock Device".to_string(),
                vec![(1, 1)],
                1024 * 1024, // 1 MB
                16 * 1024,   // 16 KB
                32,          // 32 threads
            );
            assert!(factory.arch.contains("Mock"));
            return;
        }

        let factory = KernelFactory::new(
            "NVIDIA GeForce RTX 3080".to_string(),
            vec![(8, 6)],
            10 * 1024 * 1024 * 1024, // 10 GB
            48 * 1024,               // 48 KB
            1024,                    // 1024 threads per block
        );

        // Test creating FFT kernel
        let kernel = factory
            .create_fft_kernel(1024, 0x10000, 0x20000)
            .expect("Operation failed");

        // Check configuration
        let config = kernel.config();
        assert_eq!(config.block_size, 1024);
        assert!(config.use_mixed_precision);
        assert!(config.use_tensor_cores);

        // Test creating sparse FFT kernel
        let kernel = factory
            .create_sparse_fft_kernel(
                1024,
                10,
                0x10000,
                0x20000,
                0x30000,
                SparseFFTAlgorithm::Sublinear,
                WindowFunction::Hann,
            )
            .expect("Operation failed");

        // Check configuration
        let config = kernel.config();
        assert_eq!(config.block_size, 256);
        assert!(config.use_mixed_precision);
    }

    #[test]
    fn test_kernel_launcher() {
        // Check if GPU is available
        let caps = PlatformCapabilities::detect();
        if !caps.cuda_available && !caps.gpu_available {
            // Mock test for CPU-only environments
            eprintln!("GPU not available, using mock kernel launcher test");
            let factory = KernelFactory::new(
                "Mock Device".to_string(),
                vec![(1, 1)],
                1024 * 1024,
                16 * 1024,
                32,
            );
            let launcher = KernelLauncher::new(factory);
            // Test that launcher is created successfully
            assert_eq!(launcher.get_total_memory_allocated(), 0);
            return;
        }

        let factory = KernelFactory::new(
            "NVIDIA GeForce RTX 3080".to_string(),
            vec![(8, 6)],
            10 * 1024 * 1024 * 1024, // 10 GB
            48 * 1024,               // 48 KB
            1024,                    // 1024 threads per block
        );

        let mut launcher = KernelLauncher::new(factory);

        // Test allocating memory
        let (input_address, output_address) = launcher
            .allocate_fft_memory(1024)
            .expect("Operation failed");
        assert_ne!(input_address, 0);
        assert_ne!(output_address, 0);

        // Test launching FFT kernel
        let stats = launcher
            .launch_fft_kernel(1024, input_address, output_address)
            .expect("Operation failed");

        // Check stats. The modelled time is positive, occupancy is a real model
        // output in (0, 100], and the estimated throughput is present and positive
        // for a non-trivial transform. Effective bandwidth is `None` here because
        // no real host<->device transfer takes place (no device runtime).
        assert!(stats.execution_time_ms > 0.0);
        assert!(stats.estimated_occupancy_percent > 0.0);
        assert!(stats.estimated_occupancy_percent <= 100.0);
        assert!(stats.memory_bandwidth_gb_s.is_none());
        assert!(matches!(
            stats.estimated_compute_throughput_gflops,
            Some(gflops) if gflops > 0.0
        ));

        // Test freeing memory
        launcher.free_all_memory();
        assert_eq!(launcher.get_total_memory_allocated(), 0);
    }

    #[test]
    fn test_execute_sparse_fft_kernel() {
        // Create a sparse signal
        let n = 1024;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signal = create_sparse_signal(n, &frequencies);

        // Check if GPU is available
        let caps = PlatformCapabilities::detect();
        if !caps.cuda_available && !caps.gpu_available {
            // Mock test for CPU-only environments
            eprintln!("GPU not available, using mock sparse FFT kernel test");
            // Test with mock device
            let result = execute_sparse_fft_kernel(
                &signal,
                6,
                SparseFFTAlgorithm::Sublinear,
                WindowFunction::Hann,
                "Mock Device",
                (1, 1),
                1024 * 1024, // 1 MB
            );
            // No GPU runtime: the function now computes the REAL sparse FFT on
            // the host instead of fabricating components. For a 3-tone real
            // signal the top-6 components are the 3 tones and their conjugate
            // mirrors, and the strongest tone (bin 3) must be recovered.
            let (values, indices, stats) = result.expect("Operation failed");
            assert_eq!(values.len(), 6);
            assert_eq!(indices.len(), 6);
            assert!(
                indices.contains(&3),
                "real strongest tone (bin 3) not found"
            );
            assert!(stats.execution_time_ms >= 0.0);
            return;
        }

        // Execute sparse FFT kernel with GPU
        let (values, indices, stats) = execute_sparse_fft_kernel(
            &signal,
            6,
            SparseFFTAlgorithm::Sublinear,
            WindowFunction::Hann,
            "NVIDIA GeForce RTX 3080",
            (8, 6),
            10 * 1024 * 1024 * 1024, // 10 GB
        )
        .expect("Operation failed");

        // Check results: same real computation as the CPU path.
        assert_eq!(values.len(), 6);
        assert_eq!(indices.len(), 6);
        assert!(
            indices.contains(&3),
            "real strongest tone (bin 3) not found"
        );
        assert!(stats.execution_time_ms > 0.0);
    }
}
