//! Example integration of multi-GPU synchronization with optimizers
//!
//! This module demonstrates how to use the multi-GPU synchronization primitives
//! with existing optimizers for distributed training scenarios.

#[allow(unused_imports)]
use crate::error::Result;
use crate::gpu::multi_gpu_sync::{
    create_multi_gpu_communicator, MultiGpuCommunicator, SyncFrequency,
};
use crate::optimizers::adam::Adam;

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Multi-GPU distributed optimizer wrapper
pub struct DistributedOptimizer<T: Float> {
    /// Local optimizer instance
    local_optimizer: Adam<T>,

    /// Multi-GPU communicator
    communicator: Arc<Mutex<MultiGpuCommunicator>>,

    /// Parameter registry for synchronization
    parameter_registry: HashMap<String, ParameterInfo>,

    /// Gradient buffers for accumulation
    gradient_buffers: HashMap<String, Array1<T>>,

    /// World size (total number of GPUs)
    world_size: usize,

    /// Local rank (GPU ID)
    local_rank: i32,

    /// Synchronization step counter
    sync_step: usize,

    /// Configuration
    config: DistributedConfig,
}

/// Parameter information for synchronization
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,

    /// Parameter shape
    pub shape: Vec<usize>,

    /// Total number of elements
    pub num_elements: usize,

    /// Whether this parameter needs gradient synchronization
    pub requires_sync: bool,

    /// Last synchronization step
    pub last_sync_step: usize,
}

/// Configuration for distributed optimization
#[derive(Debug, Clone)]
pub struct DistributedConfig<T: Float> {
    /// Synchronization frequency
    pub sync_frequency: SyncFrequency,

    /// Enable gradient compression
    pub enable_compression: bool,

    /// Gradient clipping threshold
    pub gradient_clip_threshold: Option<T>,

    /// Enable all-reduce optimization
    pub enable_allreduce_optimization: bool,

    /// Bucket size for gradient bucketing (elements)
    pub bucket_size: usize,

    /// Enable overlap of computation and communication
    pub enable_overlap: bool,
}

impl<T: Float + Send + Sync> DistributedOptimizer<T> {
    /// Create new distributed optimizer
    pub fn new(
        local_optimizer: Adam<T>,
        world_size: usize,
        local_rank: i32,
        config: DistributedConfig,
    ) -> Result<Self> {
        let communicator = create_multi_gpu_communicator(world_size, local_rank)?;

        Ok(Self {
            local_optimizer,
            communicator: Arc::new(Mutex::new(communicator)),
            parameter_registry: HashMap::new(),
            gradient_buffers: HashMap::new(),
            world_size,
            local_rank,
            sync_step: 0,
            config,
        })
    }

    /// Register parameters for distributed training
    pub fn register_parameters(&mut self, params: &HashMap<String, Array1<T>>) -> Result<()> {
        let mut communicator = self.communicator.lock().unwrap();

        for (name, param) in params {
            let param_info = ParameterInfo {
                name: name.clone(),
                shape: param.shape().to_vec(),
                num_elements: param.len(),
                requires_sync: true,
                last_sync_step: 0,
            };

            // Register parameter buffer for synchronization
            communicator.register_parameter_buffer::<T>(name, param.shape())?;

            // Register gradient buffer for reduction
            communicator.register_gradient_buffer::<T>(name, param.shape())?;

            // Initialize local gradient buffer
            self.gradient_buffers
                .insert(name.clone(), Array1::zeros(param.len()));

            self.parameter_registry.insert(name.clone(), param_info);
        }

        Ok(())
    }

    /// Perform distributed optimization step
    pub fn step(
        &mut self,
        params: &mut HashMap<String, Array1<T>>,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<()> {
        // Step 1: Accumulate gradients locally
        self.accumulate_gradients(gradients)?;

        // Step 2: Perform gradient synchronization if needed
        let should_sync = self.should_synchronize();
        if should_sync {
            self.synchronize_gradients()?;
        }

        // Step 3: Apply local optimizer step
        for (name, param) in params.iter_mut() {
            if let Some(gradient) = self.gradient_buffers.get(name) {
                // Scale gradients by world size for averaging
                let scaled_gradient = gradient.mapv(|x| x / num_traits::cast::cast(self.world_size).unwrap_or_else(|| T::zero()));

                // Apply optimizer step
                self.local_optimizer
                    .step_parameter(param, &scaled_gradient)?;
            }
        }

        // Step 4: Synchronize parameters if needed
        if should_sync {
            self.synchronize_parameters(params)?;
            self.sync_step += 1;
        }

        // Step 5: Clear gradient buffers
        self.clear_gradient_buffers();

        Ok(())
    }

    /// Perform asynchronous distributed optimization step with overlap
    pub fn async_step(
        &mut self,
        params: &mut HashMap<String, Array1<T>>,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<()> {
        if !self.config.enable_overlap {
            return self.step(params, gradients);
        }

        // Step 1: Start asynchronous gradient synchronization
        self.accumulate_gradients(gradients)?;

        if self.should_synchronize() {
            self.async_synchronize_gradients()?;
        }

        // Step 2: Continue with local computation while communication happens
        // (In a real implementation, this would overlap with GPU kernel execution)

        // Step 3: Wait for gradient synchronization to complete
        if self.should_synchronize() {
            self.wait_for_gradient_sync()?;
        }

        // Step 4: Apply optimizer step and parameter synchronization
        self.apply_optimizer_step(params)?;

        if self.should_synchronize() {
            self.synchronize_parameters(params)?;
            self.sync_step += 1;
        }

        self.clear_gradient_buffers();
        Ok(())
    }

    /// Broadcast parameters from master to all workers
    pub fn broadcast_parameters_from_master(
        &mut self,
        params: &mut HashMap<String, Array1<T>>,
        master_rank: i32,
    ) -> Result<()> {
        let param_names: Vec<&str> = params.keys().map(|s| s.as_str()).collect();

        {
            let mut communicator = self.communicator.lock().unwrap();
            communicator.broadcast_parameters(&param_names, master_rank)?;
        }

        // Copy parameters from GPU buffers back to local arrays
        // (In a real implementation, this would involve actual GPU memory transfers)

        Ok(())
    }

    /// Get communication performance metrics
    pub fn get_communication_metrics(&self) -> Result<String> {
        let communicator = self.communicator.lock().unwrap();
        let metrics = communicator.get_performance_metrics();

        Ok(format!(
            "Multi-GPU Communication Metrics\n\
             ================================\n\
             Total Operations: {}\n\
             Total Bytes Transferred: {:.2} MB\n\
             Average Bandwidth: {:.2} GB/s\n\
             Average Latency: {:.2} μs\n\
             Compression Ratio: {:.2}x\n\
             Overlap Efficiency: {:.2}%\n",
            metrics.total_operations,
            metrics.total_bytes_transferred as f64 / (1024.0 * 1024.0),
            metrics.average_bandwidth / 1e9,
            metrics.average_latency_us,
            metrics.compression_ratio,
            metrics.overlap_efficiency * 100.0,
        ))
    }

    /// Enable gradient compression for bandwidth optimization
    pub fn enable_gradient_compression(&mut self) -> Result<()> {
        let mut communicator = self.communicator.lock().unwrap();
        communicator.enable_gradient_compression(
            crate::gpu::multi_gpu_sync::CompressionAlgorithm::Quantization,
            6, // Compression level
        );
        Ok(())
    }

    /// Private helper methods

    fn accumulate_gradients(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        for (name, gradient) in gradients {
            if let Some(buffer) = self.gradient_buffers.get_mut(name) {
                // Accumulate gradients (simple addition for now)
                *buffer = buffer.clone() + gradient;
            }
        }
        Ok(())
    }

    fn should_synchronize(&self) -> bool {
        match self.config.sync_frequency {
            SyncFrequency::EveryStep => true,
            SyncFrequency::EveryNSteps(n) => self.sync_step % n == 0,
            SyncFrequency::Adaptive(_threshold) => {
                // In a real implementation, would check gradient change threshold
                true
            }
        }
    }

    fn synchronize_gradients(&mut self) -> Result<()> {
        let gradient_names: Vec<&str> = self.gradient_buffers.keys().map(|s| s.as_str()).collect();

        {
            let mut communicator = self.communicator.lock().unwrap();

            // Apply gradient clipping if configured
            if let Some(threshold) = self.config.gradient_clip_threshold {
                self.clip_gradients(threshold);
            }

            // Perform all-reduce on gradients
            communicator.all_reduce_gradients(&gradient_names)?;
        }

        Ok(())
    }

    fn async_synchronize_gradients(&mut self) -> Result<()> {
        let gradient_names: Vec<&str> = self.gradient_buffers.keys().map(|s| s.as_str()).collect();

        {
            let mut communicator = self.communicator.lock().unwrap();

            // Apply gradient clipping if configured
            if let Some(threshold) = self.config.gradient_clip_threshold {
                self.clip_gradients(threshold);
            }

            // Start asynchronous all-reduce
            communicator.async_all_reduce_gradients(&gradient_names)?;
        }

        Ok(())
    }

    fn wait_for_gradient_sync(&mut self) -> Result<()> {
        let mut communicator = self.communicator.lock().unwrap();
        communicator.synchronize_all_operations()?;
        Ok(())
    }

    fn apply_optimizer_step(&mut self, params: &mut HashMap<String, Array1<T>>) -> Result<()> {
        for (name, param) in params.iter_mut() {
            if let Some(gradient) = self.gradient_buffers.get(name) {
                // Scale gradients by world size for averaging
                let scaled_gradient = gradient.mapv(|x| x / num_traits::cast::cast(self.world_size).unwrap_or_else(|| T::zero()));

                // Apply optimizer step
                self.local_optimizer
                    .step_parameter(param, &scaled_gradient)?;
            }
        }
        Ok(())
    }

    fn synchronize_parameters(&mut self, params: &HashMap<String, Array1<T>>) -> Result<()> {
        let param_names: Vec<&str> = params.keys().map(|s| s.as_str()).collect();

        {
            let mut communicator = self.communicator.lock().unwrap();

            // Broadcast parameters from rank 0 to ensure consistency
            communicator.broadcast_parameters(&param_names, 0)?;
        }

        Ok(())
    }

    fn clip_gradients(&mut self, threshold: T) {
        for (_, gradient) in self.gradient_buffers.iter_mut() {
            let grad_norm = gradient.mapv(|x| x * x).sum().sqrt();

            if grad_norm > threshold {
                let scale_factor = threshold / grad_norm;
                gradient.mapv_inplace(|x| x * scale_factor);
            }
        }
    }

    fn clear_gradient_buffers(&mut self) {
        for (_, buffer) in self.gradient_buffers.iter_mut() {
            buffer.fill(T::zero());
        }
    }
}

impl<T: Float> Default for DistributedConfig<T> {
    fn default() -> Self {
        Self {
            sync_frequency: SyncFrequency::EveryStep,
            enable_compression: false,
            gradient_clip_threshold: None,
            enable_allreduce_optimization: true,
            bucket_size: 25 * 1024 * 1024, // 25M elements
            enable_overlap: true,
        }
    }
}

/// Example usage of distributed optimizer
#[allow(dead_code)]
pub fn distributed_training_example() -> Result<()> {
    // Simulation parameters
    let world_size = 4;
    let local_rank = 0;
    let param_size = 1000;

    // Create local Adam optimizer
    let adam_optimizer = Adam::new_with_config(
        0.001f32, // learning_rate
        0.9f32,   // beta1
        0.999f32, // beta2
        1e-8f32,  // epsilon
        0.0f32,   // weight_decay
    );
    let local_optimizer = adam_optimizer;

    // Create distributed optimizer
    let dist_config = DistributedConfig {
        sync_frequency: SyncFrequency::EveryStep,
        enable_compression: true,
        gradient_clip_threshold: Some(1.0),
        enable_allreduce_optimization: true,
        bucket_size: 1024 * 1024,
        enable_overlap: true,
    };

    let mut distributed_optimizer =
        DistributedOptimizer::new(local_optimizer, world_size, local_rank, dist_config)?;

    // Initialize parameters
    let mut params = HashMap::new();
    params.insert("layer1_weights".to_string(), Array1::zeros(param_size));
    params.insert("layer1_bias".to_string(), Array1::zeros(100));
    params.insert("layer2_weights".to_string(), Array1::zeros(param_size));

    // Register parameters for distributed training
    distributed_optimizer.register_parameters(&params)?;

    // Initialize parameters from master rank
    distributed_optimizer.broadcast_parameters_from_master(&mut params, 0)?;

    // Enable gradient compression
    distributed_optimizer.enable_gradient_compression()?;

    // Simulate training loop
    for step in 0..100 {
        // Simulate gradients (normally computed during forward/backward pass)
        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1_weights".to_string(),
            Array1::ones(param_size) * 0.01,
        );
        gradients.insert("layer1_bias".to_string(), Array1::ones(100) * 0.001);
        gradients.insert(
            "layer2_weights".to_string(),
            Array1::ones(param_size) * 0.005,
        );

        // Perform distributed optimization step
        if step % 10 == 0 {
            // Use synchronous step for checkpointing
            distributed_optimizer.step(&mut params, &gradients)?;
        } else {
            // Use asynchronous step for better performance
            distributed_optimizer.async_step(&mut params, &gradients)?;
        }

        // Print metrics every 20 steps
        if step % 20 == 0 {
            println!("Step {}", step);
            println!("{}", distributed_optimizer.get_communication_metrics()?);
        }
    }

    Ok(())
}

/// Benchmark multi-GPU communication performance
#[allow(dead_code)]
pub fn benchmark_multi_gpu_communication() -> Result<()> {
    let world_size = 8;
    let local_rank = 0;

    let mut communicator = create_multi_gpu_communicator(world_size, local_rank)?;

    // Test different tensor sizes
    let test_sizes = vec![1024, 10240, 102400, 1024000, 10240000];

    println!("Multi-GPU Communication Benchmark");
    println!("==================================");

    for &size in &test_sizes {
        let shape = vec![size];

        // Register test buffers
        let param_name = format!("test_param_{}", size);
        let grad_name = format!("test_grad_{}", size);

        communicator.register_parameter_buffer::<f32>(&param_name, &shape)?;
        communicator.register_gradient_buffer::<f32>(&grad_name, &shape)?;

        // Benchmark all-reduce
        let start_time = std::time::Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            communicator.all_reduce_gradients(&[&grad_name])?;
        }

        let avg_time = start_time.elapsed().as_secs_f64() / iterations as f64;
        let bandwidth = (size * 4) as f64 / avg_time / 1e9; // GB/s

        println!(
            "Size: {:>8} elements, Time: {:>8.3} ms, Bandwidth: {:>6.2} GB/s",
            size,
            avg_time * 1000.0,
            bandwidth
        );
    }

    // Print final metrics
    let metrics = communicator.get_performance_metrics();
    println!("\nFinal Communication Metrics:");
    println!("Total Operations: {}", metrics.total_operations);
    println!(
        "Average Bandwidth: {:.2} GB/s",
        metrics.average_bandwidth / 1e9
    );
    println!("Average Latency: {:.2} μs", metrics.average_latency_us);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_optimizer_creation() {
        let local_optimizer = Adam::new(0.001f32);
        let dist_config = DistributedConfig::default();

        let distributed_optimizer = DistributedOptimizer::new(
            local_optimizer,
            2, // world_size
            0, // local_rank
            dist_config,
        );

        assert!(distributed_optimizer.is_ok());
    }

    #[test]
    fn test_parameter_registration() {
        let local_optimizer = Adam::new(0.001f32);
        let dist_config = DistributedConfig::default();

        let mut distributed_optimizer = DistributedOptimizer::new(
            local_optimizer,
            2, // world_size
            0, // local_rank
            dist_config,
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("test_param".to_string(), Array1::zeros(100));

        let result = distributed_optimizer.register_parameters(&params);
        assert!(result.is_ok());

        // Check that parameter was registered
        assert!(distributed_optimizer
            .parameter_registry
            .contains_key("test_param"));
        assert!(distributed_optimizer
            .gradient_buffers
            .contains_key("test_param"));
    }

    #[test]
    fn test_sync_frequency_logic() {
        let local_optimizer = Adam::new(0.001f32);

        // Test EveryStep
        let dist_config = DistributedConfig {
            sync_frequency: SyncFrequency::EveryStep,
            ..DistributedConfig::default()
        };
        let distributed_optimizer =
            DistributedOptimizer::new(local_optimizer.clone(), 2, 0, dist_config).unwrap();
        assert!(distributed_optimizer.should_synchronize());

        // Test EveryNSteps
        let dist_config = DistributedConfig {
            sync_frequency: SyncFrequency::EveryNSteps(5),
            ..DistributedConfig::default()
        };
        let mut distributed_optimizer =
            DistributedOptimizer::new(local_optimizer.clone(), 2, 0, dist_config).unwrap();
        assert!(distributed_optimizer.should_synchronize()); // step 0
        distributed_optimizer.sync_step = 5;
        assert!(distributed_optimizer.should_synchronize()); // step 5
        distributed_optimizer.sync_step = 3;
        assert!(!distributed_optimizer.should_synchronize()); // step 3
    }
}
