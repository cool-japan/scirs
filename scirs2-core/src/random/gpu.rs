//! # GPU-Accelerated Random Number Generation
//!
//! This module provides GPU-accelerated random number generation capabilities
//! using CUDA, OpenCL, and other GPU compute backends for high-performance
//! scientific computing applications.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::gpu::{GpuContext, GpuKernelHandle};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Error types for GPU random number generation
#[derive(Debug, thiserror::Error)]
pub enum GpuRngError {
    /// GPU device not available
    #[error("GPU device not available: {0}")]
    DeviceNotAvailable(String),

    /// Memory allocation error
    #[error("GPU memory allocation error: {0}")]
    MemoryError(String),

    /// Kernel execution error
    #[error("GPU kernel execution error: {0}")]
    KernelError(String),

    /// Unsupported distribution
    #[error("Unsupported distribution for GPU generation: {0}")]
    UnsupportedDistribution(String),

    /// Invalid parameters
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

impl From<GpuRngError> for CoreError {
    fn from(err: GpuRngError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

/// GPU random number generator state
#[derive(Debug, Clone)]
pub struct GpuRngState {
    /// Generator seeds for each thread/work item
    pub seeds: Vec<u64>,
    /// Current state counters
    pub counters: Vec<u64>,
    /// Generator type
    pub generator_type: GpuGeneratorType,
}

/// Types of GPU random number generators
#[derive(Debug, Clone, PartialEq)]
pub enum GpuGeneratorType {
    /// Linear Congruential Generator (fast, lower quality)
    LinearCongruential,
    /// XorShift generator (good balance)
    XorShift,
    /// Mersenne Twister (high quality, slower)
    MersenneTwister,
    /// Philox counter-based generator
    Philox,
    /// CURAND-compatible generator
    CurandMtgp32,
}

/// Distribution types supported on GPU
#[derive(Debug, Clone)]
pub enum GpuDistribution {
    /// Uniform distribution [0, 1)
    Uniform,
    /// Normal distribution with mean and std dev
    Normal { mean: f32, std_dev: f32 },
    /// Log-normal distribution
    LogNormal { mean: f32, std_dev: f32 },
    /// Exponential distribution
    Exponential { lambda: f32 },
    /// Gamma distribution
    Gamma { alpha: f32, beta: f32 },
    /// Poisson distribution
    Poisson { lambda: f32 },
}

/// GPU random number generator
pub struct GpuRandomGenerator {
    device: Arc<GpuContext>,
    state: Arc<Mutex<GpuRngState>>,
    kernels: HashMap<String, Arc<GpuKernelHandle>>,
    work_group_size: usize,
}

impl GpuRandomGenerator {
    /// Create a new GPU random number generator
    pub fn new(device: Arc<GpuContext>, generator_type: GpuGeneratorType) -> CoreResult<Self> {
        let work_group_size = 256; // Default work group size

        // Initialize seeds using system entropy
        let num_threads = work_group_size * 32; // Multiple work groups
        let seeds = Self::generate_seeds(num_threads)?;
        let counters = vec![0u64; num_threads];

        let state = Arc::new(Mutex::new(GpuRngState {
            seeds,
            counters,
            generator_type: generator_type.clone(),
        }));

        let mut generator = Self {
            device,
            state,
            kernels: HashMap::new(),
            work_group_size,
        };

        // Compile kernels for the generator type
        generator.compile_kernels(&generator_type)?;

        Ok(generator)
    }

    /// Generate system entropy-based seeds
    fn generate_seeds(count: usize) -> CoreResult<Vec<u64>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let base_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(e.to_string()))
            })?
            .as_nanos() as u64;

        let mut seeds = Vec::with_capacity(count);
        for i in 0..count {
            let mut hasher = DefaultHasher::new();
            base_seed.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            seeds.push(hasher.finish());
        }

        Ok(seeds)
    }

    /// Compile GPU kernels for the generator type
    fn compile_kernels(&mut self, generator_type: &GpuGeneratorType) -> CoreResult<()> {
        match generator_type {
            GpuGeneratorType::XorShift => {
                self.compile_xorshift_kernels()?;
            }
            GpuGeneratorType::LinearCongruential => {
                self.compile_lcg_kernels()?;
            }
            GpuGeneratorType::Philox => {
                self.compile_philox_kernels()?;
            }
            _ => {
                return Err(CoreError::ComputationError(
                    crate::error::ErrorContext::new(format!(
                        "Generator type {:?} not yet implemented",
                        generator_type
                    )),
                ));
            }
        }

        Ok(())
    }

    /// Compile XorShift generator kernels
    fn compile_xorshift_kernels(&mut self) -> CoreResult<()> {
        // XorShift uniform random number generator kernel
        let uniform_kernel_source = r#"
        __kernel void xorshift_uniform(
            __global ulong* seeds,
            __global ulong* counters,
            __global float* output,
            const uint count
        ) {
            const uint gid = get_global_id(0);
            if (gid >= count) return;
            
            ulong seed = seeds[gid];
            
            // XorShift64 algorithm
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            
            seeds[gid] = seed;
            counters[gid]++;
            
            // Convert to [0, 1) range
            output[gid] = (float)(seed >> 11) * (1.0f / 9007199254740992.0f);
        }
        "#;

        let kernel = self
            .device
            .execute(|compiler| compiler.compile(uniform_kernel_source))
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;
        self.kernels.insert("uniform".to_string(), Arc::new(kernel));

        // Normal distribution using Box-Muller transform
        let normal_kernel_source = r#"
        __kernel void xorshift_normal(
            __global ulong* seeds,
            __global ulong* counters,
            __global float* output,
            const uint count,
            const float mean,
            const float std_dev
        ) {
            const uint gid = get_global_id(0);
            if (gid >= count / 2) return;
            
            const uint idx1 = gid * 2;
            const uint idx2 = idx1 + 1;
            
            if (idx2 >= count) return;
            
            ulong seed1 = seeds[idx1];
            ulong seed2 = seeds[idx2];
            
            // Generate two uniform random numbers
            seed1 ^= seed1 << 13; seed1 ^= seed1 >> 7; seed1 ^= seed1 << 17;
            seed2 ^= seed2 << 13; seed2 ^= seed2 >> 7; seed2 ^= seed2 << 17;
            
            seeds[idx1] = seed1;
            seeds[idx2] = seed2;
            counters[idx1]++;
            counters[idx2]++;
            
            float u1 = (float)(seed1 >> 11) * (1.0f / 9007199254740992.0f);
            float u2 = (float)(seed2 >> 11) * (1.0f / 9007199254740992.0f);
            
            // Box-Muller transform
            float r = sqrt(-2.0f * log(u1 + 1e-7f));
            float theta = 2.0f * M_PI * u2;
            
            output[idx1] = mean + std_dev * r * cos(theta);
            output[idx2] = mean + std_dev * r * sin(theta);
        }
        "#;

        let normal_kernel = self
            .device
            .execute(|compiler| compiler.compile(normal_kernel_source))
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;
        self.kernels
            .insert("normal".to_string(), Arc::new(normal_kernel));

        Ok(())
    }

    /// Compile Linear Congruential Generator kernels
    fn compile_lcg_kernels(&mut self) -> CoreResult<()> {
        let uniform_kernel_source = r#"
        __kernel void lcg_uniform(
            __global ulong* seeds,
            __global ulong* counters,
            __global float* output,
            const uint count
        ) {
            const uint gid = get_global_id(0);
            if (gid >= count) return;
            
            ulong seed = seeds[gid];
            
            // LCG: X_{n+1} = (a * X_n + c) mod m
            // Using constants from Numerical Recipes
            const ulong a = 1664525UL;
            const ulong c = 1013904223UL;
            
            seed = a * seed + c;
            seeds[gid] = seed;
            counters[gid]++;
            
            // Convert to [0, 1) range
            output[gid] = (float)(seed >> 32) * (1.0f / 4294967296.0f);
        }
        "#;

        let kernel = self
            .device
            .execute(|compiler| compiler.compile(uniform_kernel_source))
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;
        self.kernels.insert("uniform".to_string(), Arc::new(kernel));

        Ok(())
    }

    /// Compile Philox counter-based generator kernels
    fn compile_philox_kernels(&mut self) -> CoreResult<()> {
        let uniform_kernel_source = r#"
        __kernel void philox_uniform(
            __global ulong* seeds,
            __global ulong* counters,
            __global float* output,
            const uint count
        ) {
            const uint gid = get_global_id(0);
            if (gid >= count) return;
            
            ulong counter = counters[gid];
            ulong key = seeds[gid];
            
            // Simplified Philox-4x32 round function
            uint c0 = (uint)(counter);
            uint c1 = (uint)(counter >> 32);
            uint k0 = (uint)(key);
            uint k1 = (uint)(key >> 32);
            
            // Philox round constants
            const uint M0 = 0xD2511F53;
            const uint M1 = 0xCD9E8D57;
            const uint W0 = 0x9E3779B9;
            const uint W1 = 0xBB67AE85;
            
            // Perform several rounds
            for (int round = 0; round < 4; round++) {
                uint h0 = M0 * c0;
                uint h1 = M1 * c1;
                
                uint l0 = h0 ^ k0 ^ c1;
                uint l1 = h1 ^ k1 ^ c0;
                
                c0 = l0;
                c1 = l1;
                k0 += W0;
                k1 += W1;
            }
            
            counters[gid] = counter + 1;
            
            // Use high bits for better quality
            output[gid] = (float)(c0 >> 8) * (1.0f / 16777216.0f);
        }
        "#;

        let kernel = self
            .device
            .execute(|compiler| compiler.compile(uniform_kernel_source))
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;
        self.kernels.insert("uniform".to_string(), Arc::new(kernel));

        Ok(())
    }

    /// Generate random numbers on GPU
    pub fn generate(
        &self,
        distribution: &GpuDistribution,
        count: usize,
    ) -> CoreResult<Array<f32, IxDyn>> {
        match distribution {
            GpuDistribution::Uniform => self.generate_uniform(count),
            GpuDistribution::Normal { mean, std_dev } => {
                self.generate_normal(count, *mean, *std_dev)
            }
            GpuDistribution::Exponential { lambda } => self.generate_exponential(count, *lambda),
            _ => Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(format!(
                    "Distribution {:?} not yet implemented",
                    distribution
                )),
            )),
        }
    }

    /// Generate uniform random numbers [0, 1)
    pub fn generate_uniform(&self, count: usize) -> CoreResult<Array<f32, IxDyn>> {
        let kernel = self
            .kernels
            .get("uniform")
            .ok_or_else(|| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Uniform kernel not found".to_string(),
                ))
            })
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;

        // Create GPU buffers
        let state = self.state.lock().unwrap();
        let num_threads = state.seeds.len();

        let seeds_buffer = self.device.create_buffer_from_slice(&state.seeds);
        let counters_buffer = self.device.create_buffer_from_slice(&state.counters);
        let output_buffer = self.device.create_buffer::<f32>(count);

        drop(state); // Release lock

        // Set kernel arguments and execute
        kernel.set_buffer("seeds", &seeds_buffer);
        kernel.set_buffer("counters", &counters_buffer);
        kernel.set_buffer("output", &output_buffer);
        kernel.set_u32("count", count as u32);

        let num_work_groups = (count.div_ceil(self.work_group_size)) as u32;
        kernel.dispatch([num_work_groups, 1, 1]);

        // Read results back to CPU
        let results = output_buffer.to_vec();

        // Update state
        let updated_seeds = seeds_buffer.to_vec();
        let updated_counters = counters_buffer.to_vec();

        {
            let mut state = self.state.lock().unwrap();
            state.seeds[..num_threads.min(updated_seeds.len())]
                .copy_from_slice(&updated_seeds[..num_threads.min(updated_seeds.len())]);
            state.counters[..num_threads.min(updated_counters.len())]
                .copy_from_slice(&updated_counters[..num_threads.min(updated_counters.len())]);
        }

        Array::from_shape_vec(IxDyn(&[count]), results).map_err(|e| {
            CoreError::ShapeError(ErrorContext::new(format!(
                "Failed to create array from shape: {}",
                e
            )))
        })
    }

    /// Generate normal random numbers
    pub fn generate_normal(
        &self,
        count: usize,
        mean: f32,
        std_dev: f32,
    ) -> CoreResult<Array<f32, IxDyn>> {
        if let Some(kernel) = self.kernels.get("normal") {
            // Use direct normal generation if available
            let state = self.state.lock().unwrap();
            let num_threads = state.seeds.len();

            let seeds_buffer = self.device.create_buffer_from_slice(&state.seeds);
            let counters_buffer = self.device.create_buffer_from_slice(&state.counters);
            let output_buffer = self.device.create_buffer::<f32>(count);

            drop(state);

            kernel.set_buffer("seeds", &seeds_buffer);
            kernel.set_buffer("counters", &counters_buffer);
            kernel.set_buffer("output", &output_buffer);
            kernel.set_u32("count", count as u32);
            kernel.set_f32("mean", mean);
            kernel.set_f32("std_dev", std_dev);

            let num_work_groups = ((count / 2).div_ceil(self.work_group_size)) as u32;
            kernel.dispatch([num_work_groups, 1, 1]);

            let results = output_buffer.to_vec();

            // Update state
            let updated_seeds = seeds_buffer.to_vec();
            let updated_counters = counters_buffer.to_vec();

            {
                let mut state = self.state.lock().unwrap();
                state.seeds[..num_threads.min(updated_seeds.len())]
                    .copy_from_slice(&updated_seeds[..num_threads.min(updated_seeds.len())]);
                state.counters[..num_threads.min(updated_counters.len())]
                    .copy_from_slice(&updated_counters[..num_threads.min(updated_counters.len())]);
            }

            Ok(
                Array::from_shape_vec(IxDyn(&[count]), results).map_err(|e| {
                    CoreError::ShapeError(ErrorContext::new(format!(
                        "Failed to create array from shape: {}",
                        e
                    )))
                })?,
            )
        } else {
            // Fallback: generate uniform and transform using Box-Muller
            let uniform_samples = self.generate_uniform(count)?;
            let normal_samples = self.box_muller_transform(&uniform_samples, mean, std_dev)?;
            Ok(normal_samples)
        }
    }

    /// Generate exponential random numbers
    pub fn generate_exponential(&self, count: usize, lambda: f32) -> CoreResult<Array<f32, IxDyn>> {
        // Generate uniform samples and transform using inverse CDF
        let uniform_samples = self.generate_uniform(count)?;
        let exponential_samples: Vec<f32> = uniform_samples
            .iter()
            .map(|&u| -(-u.ln()) / lambda)
            .collect();

        Array::from_shape_vec(IxDyn(&[count]), exponential_samples)
            .map_err(|e| CoreError::ShapeError(ErrorContext::new(e.to_string())))
    }

    /// Box-Muller transformation for normal distribution
    fn box_muller_transform(
        &self,
        uniform_samples: &Array<f32, IxDyn>,
        mean: f32,
        std_dev: f32,
    ) -> CoreResult<Array<f32, IxDyn>> {
        let len = uniform_samples.len();
        let mut normal_samples = Vec::with_capacity(len);

        // Process pairs of uniform samples
        for i in (0..len).step_by(2) {
            let u1 = uniform_samples[i];
            let u2 = if i + 1 < len {
                uniform_samples[i + 1]
            } else {
                0.5 // Use a default value for odd count
            };

            // Box-Muller transform
            let r = (-2.0 * (u1 + 1e-7_f32).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            normal_samples.push(mean + std_dev * r * theta.cos());

            if i + 1 < len {
                normal_samples.push(mean + std_dev * r * theta.sin());
            }
        }

        normal_samples.truncate(len);
        Array::from_shape_vec(IxDyn(&[len]), normal_samples)
            .map_err(|e| CoreError::ShapeError(ErrorContext::new(e.to_string())))
    }

    /// Generate random array with specified shape
    pub fn generate_array<D: ndarray::Dimension>(
        &self,
        distribution: &GpuDistribution,
        shape: D,
    ) -> CoreResult<Array<f32, D>> {
        let total_size = shape.size();
        let flat_result = self.generate(distribution, total_size)?;

        // Reshape to desired dimensions
        let reshaped = flat_result.to_shape(shape).map_err(|e| {
            CoreError::ShapeError(ErrorContext::new(format!("Failed to reshape array: {}", e)))
        })?;
        Ok(reshaped.to_owned())
    }

    /// Get generator statistics
    pub fn get_statistics(&self) -> GpuRngStatistics {
        let state = self.state.lock().unwrap();
        let total_generated: u64 = state.counters.iter().sum();

        GpuRngStatistics {
            generator_type: state.generator_type.clone(),
            num_threads: state.seeds.len(),
            total_numbers_generated: total_generated,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let state = self.state.lock().unwrap();
        let state_size = state.seeds.len() * std::mem::size_of::<u64>() * 2; // seeds + counters
        let kernel_size = self.kernels.len() * 1024; // Rough estimate for kernel storage
        state_size + kernel_size
    }

    /// Reset generator state
    pub fn reset(&self) -> CoreResult<()> {
        let mut state = self.state.lock().unwrap();

        // Generate new seeds
        state.seeds = Self::generate_seeds(state.seeds.len())?;
        state.counters.fill(0);

        Ok(())
    }

    /// Seed the generator with a specific value
    pub fn seed(&self, base_seed: u64) -> CoreResult<()> {
        let mut state = self.state.lock().unwrap();

        for (i, seed) in state.seeds.iter_mut().enumerate() {
            *seed = base_seed.wrapping_add(i as u64);
        }
        state.counters.fill(0);

        Ok(())
    }
}

/// Statistics for GPU random number generation
#[derive(Debug, Clone)]
pub struct GpuRngStatistics {
    /// Type of generator being used
    pub generator_type: GpuGeneratorType,
    /// Number of parallel threads/work items
    pub num_threads: usize,
    /// Total numbers generated across all threads
    pub total_numbers_generated: u64,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
}

/// GPU random number manager for handling multiple generators
pub struct GpuRngManager {
    generators: HashMap<String, Arc<GpuRandomGenerator>>,
    default_device: Option<Arc<GpuContext>>,
}

impl GpuRngManager {
    /// Create a new GPU RNG manager
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
            default_device: None,
        }
    }

    /// Set the default GPU device
    pub fn set_default_device(&mut self, device: Arc<GpuContext>) {
        self.default_device = Some(device);
    }

    /// Create a named generator
    pub fn create_generator(
        &mut self,
        name: &str,
        generator_type: GpuGeneratorType,
        device: Option<Arc<GpuContext>>,
    ) -> CoreResult<Arc<GpuRandomGenerator>> {
        let device = device
            .or_else(|| self.default_device.clone())
            .ok_or_else(|| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "No GPU device available".to_string(),
                ))
            })
            .map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "GPU kernel compilation failed: {}",
                    e
                )))
            })?;

        let generator = Arc::new(GpuRandomGenerator::new(device, generator_type)?);
        self.generators.insert(name.to_string(), generator.clone());

        Ok(generator)
    }

    /// Get a generator by name
    pub fn get_generator(&self, name: &str) -> Option<Arc<GpuRandomGenerator>> {
        self.generators.get(name).cloned()
    }

    /// Remove a generator
    pub fn remove_generator(&mut self, name: &str) -> bool {
        self.generators.remove(name).is_some()
    }

    /// List all generator names
    pub fn list_generators(&self) -> Vec<String> {
        self.generators.keys().cloned().collect()
    }

    /// Get overall statistics
    pub fn get_overall_statistics(&self) -> ManagerStatistics {
        let total_generators = self.generators.len();
        let total_memory: usize = self
            .generators
            .values()
            .map(|gen| gen.get_statistics().memory_usage)
            .sum();
        let total_generated: u64 = self
            .generators
            .values()
            .map(|gen| gen.get_statistics().total_numbers_generated)
            .sum();

        ManagerStatistics {
            total_generators,
            total_memory_usage: total_memory,
            total_numbers_generated: total_generated,
        }
    }
}

impl Default for GpuRngManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStatistics {
    /// Total number of generators
    pub total_generators: usize,
    /// Total memory usage across all generators
    pub total_memory_usage: usize,
    /// Total numbers generated across all generators
    pub total_numbers_generated: u64,
}

/// Convenience functions for GPU random number generation
pub mod utils {
    use super::*;

    /// Create a quick GPU uniform random array
    pub fn gpu_uniform_array(
        device: Arc<GpuContext>,
        shape: &[usize],
    ) -> CoreResult<Array<f32, IxDyn>> {
        let generator = GpuRandomGenerator::new(device, GpuGeneratorType::XorShift)?;
        let total_size = shape.iter().product();
        generator.generate(&GpuDistribution::Uniform, total_size)
    }

    /// Create a quick GPU normal random array
    pub fn gpu_normal_array(
        device: Arc<GpuContext>,
        shape: &[usize],
        mean: f32,
        std_dev: f32,
    ) -> CoreResult<Array<f32, IxDyn>> {
        let generator = GpuRandomGenerator::new(device, GpuGeneratorType::XorShift)?;
        let total_size = shape.iter().product();
        generator.generate(&GpuDistribution::Normal { mean, std_dev }, total_size)
    }

    /// Benchmark GPU vs CPU random number generation
    pub fn benchmark_gpu_vs_cpu(
        device: Arc<GpuContext>,
        count: usize,
    ) -> CoreResult<BenchmarkResults> {
        use std::time::Instant;

        // GPU benchmark
        let start = Instant::now();
        let gpu_generator = GpuRandomGenerator::new(device, GpuGeneratorType::XorShift)?;
        let _gpu_result = gpu_generator.generate(&GpuDistribution::Uniform, count)?;
        let gpu_duration = start.elapsed();

        // CPU benchmark
        let start = Instant::now();
        let mut cpu_rng = crate::random::Random::default();
        let _cpu_result: Vec<f64> = (0..count)
            .map(|_| cpu_rng.sample(rand_distr::Uniform::new(0.0, 1.0).unwrap()))
            .collect();
        let cpu_duration = start.elapsed();

        Ok(BenchmarkResults {
            count,
            gpu_duration,
            cpu_duration,
            speedup: cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64(),
        })
    }
}

/// Benchmark results for GPU vs CPU comparison
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Number of samples generated
    pub count: usize,
    /// Time taken by GPU
    pub gpu_duration: std::time::Duration,
    /// Time taken by CPU
    pub cpu_duration: std::time::Duration,
    /// Speedup factor (CPU time / GPU time)
    pub speedup: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_rng_state_creation() {
        let seeds = vec![12345, 67890, 13579];
        let counters = vec![0, 0, 0];
        let state = GpuRngState {
            seeds,
            counters,
            generator_type: GpuGeneratorType::XorShift,
        };

        assert_eq!(state.seeds.len(), 3);
        assert_eq!(state.generator_type, GpuGeneratorType::XorShift);
    }

    #[test]
    fn test_distribution_types() {
        let uniform = GpuDistribution::Uniform;
        let normal = GpuDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        };
        let exponential = GpuDistribution::Exponential { lambda: 1.0 };

        // Test that distributions can be created and compared
        assert!(matches!(uniform, GpuDistribution::Uniform));

        match normal {
            GpuDistribution::Normal { mean, std_dev } => {
                assert_eq!(mean, 0.0);
                assert_eq!(std_dev, 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }

        match exponential {
            GpuDistribution::Exponential { lambda } => {
                assert_eq!(lambda, 1.0);
            }
            _ => panic!("Expected Normal distribution"),
        }
    }

    #[test]
    fn test_gpu_rng_manager() {
        let manager = GpuRngManager::new();

        // Test initial state
        assert_eq!(manager.list_generators().len(), 0);

        // Test statistics
        let stats = manager.get_overall_statistics();
        assert_eq!(stats.total_generators, 0);
        assert_eq!(stats.total_memory_usage, 0);
        assert_eq!(stats.total_numbers_generated, 0);
    }

    #[test]
    fn test_benchmark_results() {
        use std::time::Duration;

        let results = BenchmarkResults {
            count: 1000,
            gpu_duration: Duration::from_millis(10),
            cpu_duration: Duration::from_millis(50),
            speedup: 5.0,
        };

        assert_eq!(results.count, 1000);
        assert_eq!(results.speedup, 5.0);
    }
}
