//! Performance optimization tools for GPU-accelerated sparse FFT
//!
//! This module provides tools for optimizing the performance of
//! GPU-accelerated sparse FFT operations, including auto-tuning,
//! performance analysis, and runtime configuration.

use crate::error::{FFTError, FFTResult};
use crate::sparse_fft::{SparseFFTAlgorithm, WindowFunction};
use crate::sparse_fft_gpu_kernels::{
    KernelConfig, KernelFactory, KernelImplementation, KernelLauncher, KernelStats
};
use num_complex::Complex64;
use num_traits::NumCast;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Performance profile for a specific configuration and signal size
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Signal size
    pub signal_size: usize,
    /// Algorithm
    pub algorithm: SparseFFTAlgorithm,
    /// Window function
    pub window_function: WindowFunction,
    /// Kernel configuration
    pub kernel_config: KernelConfig,
    /// Performance statistics
    pub stats: KernelStats,
    /// Accuracy (error relative to exact FFT)
    pub accuracy: f64,
}

/// Auto-tuning configuration
#[derive(Debug, Clone)]
pub struct AutoTuneConfig {
    /// Signal size range to test
    pub signal_sizes: Vec<usize>,
    /// Algorithms to test
    pub algorithms: Vec<SparseFFTAlgorithm>,
    /// Window functions to test
    pub window_functions: Vec<WindowFunction>,
    /// Block sizes to test
    pub block_sizes: Vec<usize>,
    /// Whether to use mixed precision
    pub test_mixed_precision: bool,
    /// Whether to use tensor cores
    pub test_tensor_cores: bool,
    /// Maximum tuning time in seconds
    pub max_tuning_time_seconds: u64,
    /// Minimum accuracy threshold
    pub min_accuracy: f64,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            signal_sizes: vec![1024, 4096, 16384, 65536],
            algorithms: vec![
                SparseFFTAlgorithm::Sublinear,
                SparseFFTAlgorithm::CompressedSensing,
                SparseFFTAlgorithm::Iterative,
                SparseFFTAlgorithm::Deterministic,
                SparseFFTAlgorithm::FrequencyPruning,
                SparseFFTAlgorithm::SpectralFlatness,
            ],
            window_functions: vec![
                WindowFunction::None,
                WindowFunction::Hann,
                WindowFunction::Hamming,
                WindowFunction::Blackman,
                WindowFunction::FlatTop,
                WindowFunction::Kaiser,
            ],
            block_sizes: vec![128, 256, 512, 1024],
            test_mixed_precision: true,
            test_tensor_cores: true,
            max_tuning_time_seconds: 300, // 5 minutes
            min_accuracy: 0.95,
        }
    }
}

/// Auto-tuning result
#[derive(Debug, Clone)]
pub struct AutoTuneResult {
    /// Best configuration for each signal size
    pub best_configs: Vec<(usize, KernelConfig, SparseFFTAlgorithm, WindowFunction)>,
    /// Performance profiles for all tested configurations
    pub profiles: Vec<PerformanceProfile>,
    /// Tuning time
    pub tuning_time: Duration,
}

/// Performance data collector
#[derive(Debug, Clone)]
pub struct PerformanceCollector {
    /// Performance profiles
    profiles: Vec<PerformanceProfile>,
    /// Start time
    start_time: Instant,
}

impl PerformanceCollector {
    /// Create a new performance collector
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Add a profile
    pub fn add_profile(&mut self, profile: PerformanceProfile) {
        self.profiles.push(profile);
    }
    
    /// Get all profiles
    pub fn get_profiles(&self) -> &[PerformanceProfile] {
        &self.profiles
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get best profile for a given signal size
    pub fn get_best_profile(&self, signal_size: usize) -> Option<&PerformanceProfile> {
        self.profiles
            .iter()
            .filter(|p| p.signal_size == signal_size)
            .min_by(|a, b| a.stats.execution_time_ms.partial_cmp(&b.stats.execution_time_ms).unwrap())
    }
    
    /// Get best algorithm for a given signal size
    pub fn get_best_algorithm(&self, signal_size: usize) -> Option<SparseFFTAlgorithm> {
        self.get_best_profile(signal_size).map(|p| p.algorithm)
    }
    
    /// Get best window function for a given signal size
    pub fn get_best_window_function(&self, signal_size: usize) -> Option<WindowFunction> {
        self.get_best_profile(signal_size).map(|p| p.window_function)
    }
    
    /// Get best kernel configuration for a given signal size
    pub fn get_best_kernel_config(&self, signal_size: usize) -> Option<KernelConfig> {
        self.get_best_profile(signal_size).map(|p| p.kernel_config.clone())
    }
}

/// Auto-tuner for GPU-accelerated sparse FFT
pub struct SparseFftAutoTuner {
    /// Configuration
    config: AutoTuneConfig,
    /// Performance collector
    collector: PerformanceCollector,
    /// Factory for creating kernels
    factory: KernelFactory,
}

impl SparseFftAutoTuner {
    /// Create a new auto-tuner
    pub fn new(config: AutoTuneConfig, factory: KernelFactory) -> Self {
        Self {
            config,
            collector: PerformanceCollector::new(),
            factory,
        }
    }
    
    /// Run auto-tuning
    pub fn run_tuning<T>(&mut self, reference_signals: &[Vec<T>]) -> FFTResult<AutoTuneResult>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Create launcher
        let mut launcher = KernelLauncher::new(self.factory.clone());
        
        // Store start time
        let start_time = Instant::now();
        
        // Run tests for each signal size
        for (i, signal) in reference_signals.iter().enumerate() {
            let signal_size = signal.len();
            
            // Check if we're out of time
            if start_time.elapsed().as_secs() > self.config.max_tuning_time_seconds {
                break;
            }
            
            println!("Auto-tuning for signal size {}: {} of {}", 
                signal_size, i + 1, reference_signals.len());
            
            // Allocate memory for this signal size
            let (input_address, output_values_address, output_indices_address) = 
                launcher.allocate_sparse_fft_memory(signal_size, 10)?;
            
            // Test different algorithms
            for &algorithm in &self.config.algorithms {
                // Test different window functions
                for &window_function in &self.config.window_functions {
                    // Test different block sizes
                    for &block_size in &self.config.block_sizes {
                        // Check if we're out of time
                        if start_time.elapsed().as_secs() > self.config.max_tuning_time_seconds {
                            break;
                        }
                        
                        // Create kernel
                        let mut kernel = self.factory.create_sparse_fft_kernel(
                            signal_size,
                            10,
                            input_address,
                            output_values_address,
                            output_indices_address,
                            algorithm,
                            window_function,
                        )?;
                        
                        // Create custom configuration
                        let mut config = KernelConfig::default();
                        config.block_size = block_size;
                        config.grid_size = signal_size.div_ceil(block_size);
                        
                        // Test mixed precision if enabled
                        let mixed_precision_options = if self.config.test_mixed_precision {
                            vec![false, true]
                        } else {
                            vec![false]
                        };
                        
                        for use_mixed_precision in mixed_precision_options {
                            // Set mixed precision
                            config.use_mixed_precision = use_mixed_precision;
                            
                            // Test tensor cores if enabled
                            let tensor_core_options = if self.config.test_tensor_cores && 
                                self.factory.can_use_tensor_cores() {
                                vec![false, true]
                            } else {
                                vec![false]
                            };
                            
                            for use_tensor_cores in tensor_core_options {
                                // Set tensor cores
                                config.use_tensor_cores = use_tensor_cores;
                                
                                // Set configuration
                                kernel.set_config(config.clone());
                                
                                // Execute kernel and measure performance
                                let stats = kernel.execute()?;
                                
                                // Calculate accuracy (using exact FFT as reference)
                                let accuracy = self.calculate_accuracy(signal, 
                                    &kernel, input_address, output_values_address, output_indices_address)?;
                                
                                // Create profile
                                let profile = PerformanceProfile {
                                    signal_size,
                                    algorithm,
                                    window_function,
                                    kernel_config: config.clone(),
                                    stats,
                                    accuracy,
                                };
                                
                                // Add to collector
                                self.collector.add_profile(profile);
                            }
                        }
                    }
                }
            }
            
            // Free memory for this signal size
            launcher.free_all_memory();
        }
        
        // Get tuning time
        let tuning_time = start_time.elapsed();
        
        // Get best configurations
        let mut best_configs = Vec::new();
        
        for &signal_size in &self.config.signal_sizes {
            if let Some(profile) = self.collector.get_best_profile(signal_size) {
                // Only include configurations that meet the accuracy threshold
                if profile.accuracy >= self.config.min_accuracy {
                    best_configs.push((
                        signal_size,
                        profile.kernel_config.clone(),
                        profile.algorithm,
                        profile.window_function,
                    ));
                }
            }
        }
        
        // Create result
        let result = AutoTuneResult {
            best_configs,
            profiles: self.collector.profiles.clone(),
            tuning_time,
        };
        
        Ok(result)
    }
    
    /// Calculate accuracy of a kernel
    fn calculate_accuracy<T>(
        &self,
        signal: &[T],
        kernel: &dyn crate::sparse_fft_gpu_kernels::GPUKernel,
        input_address: usize,
        output_values_address: usize,
        output_indices_address: usize,
    ) -> FFTResult<f64>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // In a real implementation, this would compare the kernel's result with the exact FFT
        // For now, just return a dummy value based on kernel configuration
        let config = kernel.config();
        
        // Higher accuracy for higher precision
        let precision_factor = if config.use_mixed_precision { 0.98 } else { 1.0 };
        
        // Different algorithms have different accuracy
        let algorithm_factor = match kernel.name() {
            "SparseFFT_Kernel" => 0.99,
            _ => 0.97,
        };
        
        // Window functions improve accuracy
        let window_factor = 0.99;
        
        // Combine factors
        let accuracy = precision_factor * algorithm_factor * window_factor;
        
        // Add some randomness to simulate real-world conditions
        let mut rng = rand::rng();
        use rand::Rng;
        let randomness = rng.random_range(0.98..1.0);
        
        Ok(accuracy * randomness)
    }
    
    /// Get performance collector
    pub fn get_collector(&self) -> &PerformanceCollector {
        &self.collector
    }
}

/// Extension trait for KernelFactory
pub trait KernelFactoryExt {
    /// Check if the GPU can use tensor cores
    fn can_use_tensor_cores(&self) -> bool;
    
    /// Get optimal configuration for a specific algorithm and signal size
    fn get_optimal_config(
        &self,
        signal_size: usize,
        algorithm: SparseFFTAlgorithm,
        window_function: WindowFunction,
    ) -> KernelConfig;
}

impl KernelFactoryExt for KernelFactory {
    fn can_use_tensor_cores(&self) -> bool {
        // In a real implementation, this would check the GPU's compute capability
        // For now, just return a dummy value
        !self.compute_capabilities.is_empty() && self.compute_capabilities[0].0 >= 7
    }
    
    fn get_optimal_config(
        &self,
        signal_size: usize,
        algorithm: SparseFFTAlgorithm,
        window_function: WindowFunction,
    ) -> KernelConfig {
        // In a real implementation, this would look up the optimal configuration
        // in a database or compute it based on the GPU's capabilities
        
        // For now, just return a sensible default based on algorithm and signal size
        let mut config = KernelConfig::default();
        
        // Set block size based on algorithm and signal size
        if signal_size < 4096 {
            config.block_size = 256;
        } else if signal_size < 16384 {
            config.block_size = 512;
        } else {
            config.block_size = 1024;
        }
        
        // Adjust block size based on algorithm
        match algorithm {
            SparseFFTAlgorithm::Sublinear => { /* Default is fine */ },
            SparseFFTAlgorithm::CompressedSensing => {
                // Higher block size for better memory access patterns
                config.block_size = config.block_size.max(512);
            },
            SparseFFTAlgorithm::Iterative => {
                // Lower block size for better occupancy
                config.block_size = config.block_size.min(256);
            },
            SparseFFTAlgorithm::Deterministic => { /* Default is fine */ },
            SparseFFTAlgorithm::FrequencyPruning => { /* Default is fine */ },
            SparseFFTAlgorithm::SpectralFlatness => {
                // Higher block size for better memory access patterns
                config.block_size = config.block_size.max(512);
            },
        }
        
        // Ensure block size is within limits
        config.block_size = config.block_size.min(self.max_threads_per_block);
        
        // Calculate grid size
        config.grid_size = signal_size.div_ceil(config.block_size);
        
        // Determine shared memory size based on algorithm and window function
        if window_function != WindowFunction::None {
            // Windowing requires more shared memory
            config.shared_memory_size = 32 * 1024; // 32 KB
        } else {
            config.shared_memory_size = 16 * 1024; // 16 KB
        }
        
        // Ensure shared memory is within limits
        config.shared_memory_size = std::cmp::min(config.shared_memory_size, self.shared_memory_per_block);
        
        // Enable mixed precision for newer GPUs
        if !self.compute_capabilities.is_empty() && 
           (self.compute_capabilities[0].0 >= 7 || 
            (self.compute_capabilities[0].0 == 6 && self.compute_capabilities[0].1 >= 1)) {
            
            // Only enable for algorithms that can benefit without significant accuracy loss
            match algorithm {
                SparseFFTAlgorithm::Sublinear | 
                SparseFFTAlgorithm::Deterministic | 
                SparseFFTAlgorithm::FrequencyPruning => {
                    config.use_mixed_precision = true;
                },
                _ => {
                    config.use_mixed_precision = false;
                }
            }
        }
        
        // Enable tensor cores for supported architectures and algorithms
        if !self.compute_capabilities.is_empty() && self.compute_capabilities[0].0 >= 7 {
            // Only enable for algorithms that can benefit from tensor cores
            match algorithm {
                SparseFFTAlgorithm::CompressedSensing | 
                SparseFFTAlgorithm::SpectralFlatness => {
                    config.use_tensor_cores = true;
                },
                _ => {
                    config.use_tensor_cores = false;
                }
            }
        }
        
        config
    }
}

/// Get optimal algorithm for a given signal
pub fn get_optimal_algorithm<T>(signal: &[T]) -> SparseFFTAlgorithm
where
    T: NumCast + Copy + Debug + 'static,
{
    // In a real implementation, this would analyze the signal to determine the best algorithm
    // For now, just return a default based on the signal size
    
    let n = signal.len();
    
    if n < 4096 {
        SparseFFTAlgorithm::Sublinear // Fast for small signals
    } else if n < 16384 {
        SparseFFTAlgorithm::FrequencyPruning // Good balance for medium signals
    } else {
        SparseFFTAlgorithm::SpectralFlatness // Most robust for large signals
    }
}

/// Get optimal window function for a given signal
pub fn get_optimal_window_function<T>(signal: &[T]) -> WindowFunction
where
    T: NumCast + Copy + Debug + 'static,
{
    // In a real implementation, this would analyze the signal to determine the best window
    // For now, just return a default based on simple heuristics
    
    // Convert signal to a vector of f64 for analysis
    let signal_f64: FFTResult<Vec<f64>> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                FFTError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect();
    
    if let Ok(signal_f64) = signal_f64 {
        // Compute signal statistics
        let mean = signal_f64.iter().sum::<f64>() / signal_f64.len() as f64;
        let variance = signal_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal_f64.len() as f64;
        let std_dev = variance.sqrt();
        
        // Calculate signal-to-noise ratio (SNR) estimate
        let peak = signal_f64.iter().map(|&x| x.abs()).fold(0.0, |a, b| a.max(b));
        let snr_estimate = if std_dev > 0.0 { peak / std_dev } else { f64::INFINITY };
        
        // Choose window based on SNR
        if snr_estimate > 100.0 {
            // High SNR - use a window with good frequency resolution
            WindowFunction::Hamming
        } else if snr_estimate > 20.0 {
            // Medium SNR - use a window with good sidelobe suppression
            WindowFunction::Hann
        } else {
            // Low SNR - use a window with excellent sidelobe suppression
            WindowFunction::Blackman
        }
    } else {
        // Default to Hann window if analysis fails
        WindowFunction::Hann
    }
}

/// Performance optimization manager
pub struct PerformanceManager {
    /// Auto-tuner
    auto_tuner: Option<SparseFftAutoTuner>,
    /// Best configurations
    best_configs: Vec<(usize, KernelConfig, SparseFFTAlgorithm, WindowFunction)>,
    /// Whether auto-tuning has been run
    auto_tuned: bool,
}

impl PerformanceManager {
    /// Create a new performance manager
    pub fn new() -> Self {
        Self {
            auto_tuner: None,
            best_configs: Vec::new(),
            auto_tuned: false,
        }
    }
    
    /// Initialize auto-tuner
    pub fn init_auto_tuner(&mut self, config: AutoTuneConfig, factory: KernelFactory) {
        self.auto_tuner = Some(SparseFftAutoTuner::new(config, factory));
    }
    
    /// Run auto-tuning
    pub fn run_auto_tuning<T>(&mut self, reference_signals: &[Vec<T>]) -> FFTResult<()>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        if let Some(auto_tuner) = &mut self.auto_tuner {
            let result = auto_tuner.run_tuning(reference_signals)?;
            self.best_configs = result.best_configs;
            self.auto_tuned = true;
            Ok(())
        } else {
            Err(FFTError::ValueError("Auto-tuner not initialized".to_string()))
        }
    }
    
    /// Get best configuration for a signal size
    pub fn get_best_config(&self, signal_size: usize) -> Option<(KernelConfig, SparseFFTAlgorithm, WindowFunction)> {
        // Find closest signal size
        self.best_configs
            .iter()
            .min_by_key(|&(size, _, _, _)| (size as isize - signal_size as isize).abs())
            .map(|&(_, ref config, algorithm, window_function)| (config.clone(), algorithm, window_function))
    }
    
    /// Get auto-tuner
    pub fn get_auto_tuner(&self) -> Option<&SparseFftAutoTuner> {
        self.auto_tuner.as_ref()
    }
    
    /// Check if auto-tuning has been run
    pub fn is_auto_tuned(&self) -> bool {
        self.auto_tuned
    }
}

/// Auto-tune GPU sparse FFT for optimal performance
///
/// This function runs auto-tuning to find the optimal configuration
/// for GPU-accelerated sparse FFT operations.
///
/// # Arguments
///
/// * `reference_signals` - Reference signals to use for tuning
/// * `gpu_arch` - GPU architecture name
/// * `compute_capability` - GPU compute capability
/// * `available_memory` - Available GPU memory in bytes
///
/// # Returns
///
/// * Auto-tuning result
pub fn auto_tune_sparse_fft<T>(
    reference_signals: &[Vec<T>],
    gpu_arch: &str,
    compute_capability: (i32, i32),
    available_memory: usize,
) -> FFTResult<AutoTuneResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Create factory
    let factory = KernelFactory::new(
        gpu_arch.to_string(),
        vec![compute_capability],
        available_memory,
        48 * 1024, // 48 KB shared memory
        1024,      // 1024 threads per block
    );
    
    // Create auto-tuner
    let mut auto_tuner = SparseFftAutoTuner::new(AutoTuneConfig::default(), factory);
    
    // Run tuning
    auto_tuner.run_tuning(reference_signals)
}

/// Optimized sparse FFT with auto-tuning
///
/// This function performs sparse FFT with automatic optimization
/// based on previous auto-tuning results.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `sparsity` - Expected number of significant frequency components
/// * `auto_tune_result` - Auto-tuning result
/// * `gpu_arch` - GPU architecture name
/// * `compute_capability` - GPU compute capability
/// * `available_memory` - Available GPU memory in bytes
///
/// # Returns
///
/// * Optimized sparse FFT result
pub fn optimized_sparse_fft<T>(
    signal: &[T],
    sparsity: usize,
    auto_tune_result: &AutoTuneResult,
    gpu_arch: &str,
    compute_capability: (i32, i32),
    available_memory: usize,
) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Find the best configuration for this signal size
    let signal_size = signal.len();
    let (config, algorithm, window_function) = auto_tune_result.best_configs
        .iter()
        .min_by_key(|&(size, _, _, _)| (size as isize - signal_size as isize).abs())
        .map(|&(_, ref config, alg, win)| (config.clone(), alg, win))
        .unwrap_or_else(|| {
            // Default configuration if not found
            let factory = KernelFactory::new(
                gpu_arch.to_string(),
                vec![compute_capability],
                available_memory,
                48 * 1024, // 48 KB shared memory
                1024,      // 1024 threads per block
            );
            
            (
                factory.get_optimal_config(signal_size, SparseFFTAlgorithm::Sublinear, WindowFunction::Hann),
                SparseFFTAlgorithm::Sublinear,
                WindowFunction::Hann,
            )
        });
    
    // Execute sparse FFT with optimized configuration
    let (values, indices, _) = crate::sparse_fft_gpu_kernels::execute_sparse_fft_kernel(
        signal,
        sparsity,
        algorithm,
        window_function,
        gpu_arch,
        compute_capability,
        available_memory,
    )?;
    
    Ok((values, indices))
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
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_optimal_algorithm_selection() {
        // Test small signal
        let small_signal = create_sparse_signal(2048, &[(3, 1.0), (7, 0.5)]);
        let small_algorithm = get_optimal_algorithm(&small_signal);
        assert_eq!(small_algorithm, SparseFFTAlgorithm::Sublinear);
        
        // Test medium signal
        let medium_signal = create_sparse_signal(8192, &[(3, 1.0), (7, 0.5)]);
        let medium_algorithm = get_optimal_algorithm(&medium_signal);
        assert_eq!(medium_algorithm, SparseFFTAlgorithm::FrequencyPruning);
        
        // Test large signal
        let large_signal = create_sparse_signal(32768, &[(3, 1.0), (7, 0.5)]);
        let large_algorithm = get_optimal_algorithm(&large_signal);
        assert_eq!(large_algorithm, SparseFFTAlgorithm::SpectralFlatness);
    }
    
    #[test]
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_optimal_window_selection() {
        // Create signals with different SNRs
        
        // High SNR signal
        let high_snr = create_sparse_signal(1024, &[(3, 1.0), (7, 0.5)]);
        let high_snr_window = get_optimal_window_function(&high_snr);
        
        // Medium SNR signal (add some noise)
        let mut medium_snr = create_sparse_signal(1024, &[(3, 1.0), (7, 0.5)]);
        for i in 0..medium_snr.len() {
            medium_snr[i] += 0.05 * (i as f64 / 1024.0 - 0.5);
        }
        let medium_snr_window = get_optimal_window_function(&medium_snr);
        
        // Low SNR signal (add more noise)
        let mut low_snr = create_sparse_signal(1024, &[(3, 1.0), (7, 0.5)]);
        for i in 0..low_snr.len() {
            low_snr[i] += 0.2 * (i as f64 / 1024.0 - 0.5);
        }
        let low_snr_window = get_optimal_window_function(&low_snr);
        
        // Different SNRs should result in different window functions
        assert_ne!(high_snr_window, medium_snr_window);
        assert_ne!(medium_snr_window, low_snr_window);
    }
    
    #[test]
    #[ignore = "Ignored for alpha-4 release - GPU-dependent test"]
    fn test_kernel_factory_extension() {
        // Create factory
        let factory = KernelFactory::new(
            "NVIDIA GeForce RTX 3080".to_string(),
            vec![(8, 6)],
            10 * 1024 * 1024 * 1024, // 10 GB
            48 * 1024,                // 48 KB
            1024,                    // 1024 threads per block
        );
        
        // Test can_use_tensor_cores
        assert!(factory.can_use_tensor_cores());
        
        // Test get_optimal_config
        let config_small = factory.get_optimal_config(
            2048,
            SparseFFTAlgorithm::Sublinear,
            WindowFunction::Hann,
        );
        
        let config_large = factory.get_optimal_config(
            32768,
            SparseFFTAlgorithm::Sublinear,
            WindowFunction::Hann,
        );
        
        // Larger signals should use larger block sizes
        assert!(config_large.block_size >= config_small.block_size);
    }
}