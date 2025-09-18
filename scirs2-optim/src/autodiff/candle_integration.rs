//! Candle-core integration for automatic differentiation
//!
//! This module provides seamless integration with the candle-core tensor library
//! for automatic gradient computation in optimization algorithms.

use ndarray::{Array, Array1, Array2, ArrayBase, Dimension, Ix1, Ix2};
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::autodiff::{AutodiffEngine, AutodiffConfig, Variable, Operation};
use crate::error::{OptimError, Result};

/// Candle tensor wrapper for automatic differentiation
pub struct CandleTensor<T: Float> {
    /// Underlying data
    data: Array1<T>,
    
    /// Gradient information
    grad: Option<Array1<T>>,
    
    /// Whether this tensor requires gradients
    requires_grad: bool,
    
    /// Shape information
    shape: Vec<usize>,
    
    /// Data type
    dtype: CandleDataType,
    
    /// Device information
    device: CandleDevice,
    
    /// Computational graph node ID
    node_id: Option<usize>,
    
    /// Reference to autodiff engine
    engine: Option<Arc<Mutex<AutodiffEngine<T>>>>,
}

/// Candle-compatible data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CandleDataType {
    F16,
    F32,
    F64,
    BF16,
    U8,
    U32,
    I64,
}

/// Candle device types
#[derive(Debug, Clone, PartialEq)]
pub enum CandleDevice {
    Cpu,
    Cuda(usize), // GPU device ID
    Metal(usize),
}

/// Candle optimizer integration
pub struct CandleOptimizer<T: Float> {
    /// AutoDiff engine
    engine: Arc<Mutex<AutodiffEngine<T>>>,
    
    /// Tensor registry
    tensors: HashMap<String, CandleTensor<T>>,
    
    /// Optimizer configuration
    config: CandleOptimizerConfig,
    
    /// Current device
    device: CandleDevice,
    
    /// Gradient accumulation buffer
    grad_buffer: HashMap<String, Array1<T>>,
    
    /// Mixed precision settings
    mixed_precision: MixedPrecisionConfig,
}

/// Configuration for Candle optimizer integration
#[derive(Debug, Clone)]
pub struct CandleOptimizerConfig {
    /// Enable automatic mixed precision
    pub auto_mixed_precision: bool,
    
    /// Loss scaling for mixed precision
    pub loss_scale: f32,
    
    /// Dynamic loss scaling
    pub dynamic_loss_scaling: bool,
    
    /// Gradient clipping threshold
    pub gradient_clip_value: Option<f32>,
    
    /// Gradient clipping norm type
    pub gradient_clip_norm: Option<f32>,
    
    /// Enable gradient accumulation
    pub gradient_accumulation: bool,
    
    /// Accumulation steps
    pub accumulation_steps: usize,
    
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Checkpointing segments
    pub checkpoint_segments: usize,
    
    /// Enable tensor fusion
    pub enable_fusion: bool,
    
    /// Memory optimization level
    pub memory_optimization: MemoryOptimizationLevel,
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision
    pub enabled: bool,
    
    /// Forward pass precision
    pub forward_precision: CandleDataType,
    
    /// Backward pass precision  
    pub backward_precision: CandleDataType,
    
    /// Parameter precision
    pub param_precision: CandleDataType,
    
    /// Optimizer state precision
    pub optimizer_precision: CandleDataType,
    
    /// Loss scaling strategy
    pub loss_scaling: LossScalingStrategy,
}

/// Loss scaling strategies
#[derive(Debug, Clone)]
pub enum LossScalingStrategy {
    /// Fixed scaling factor
    Fixed(f32),
    
    /// Dynamic scaling with growth/reduction factors
    Dynamic {
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    },
    
    /// Adaptive scaling based on gradient statistics
    Adaptive {
        target_gradient_norm: f32,
        adaptation_rate: f32,
    },
}

/// Memory optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryOptimizationLevel {
    /// No memory optimization
    None,
    
    /// Basic memory optimization
    Basic,
    
    /// Aggressive memory optimization
    Aggressive,
    
    /// Maximum memory optimization (may impact performance)
    Maximum,
}

impl<T: Float + Default + Clone> CandleTensor<T> {
    /// Create a new Candle tensor
    pub fn new(
        data: Array1<T>,
        requires_grad: bool,
        dtype: CandleDataType,
        device: CandleDevice,
    ) -> Self {
        let shape = vec![data.len()];
        
        Self {
            data_grad: None,
            requires_grad,
            shape,
            dtype,
            device,
            node_id: None,
            engine: None,
        }
    }
    
    /// Create tensor from shape and fill value
    pub fn full(
        shape: &[usize],
        fill_value: T,
        dtype: CandleDataType,
        device: CandleDevice,
    ) -> Self {
        let size = shape.iter().product();
        let data = Array1::from_elem(size, fill_value);
        
        Self::new(data, false, dtype, device)
    }
    
    /// Create tensor with zeros
    pub fn zeros(shape: &[usize], dtype: CandleDataType, device: CandleDevice) -> Self {
        Self::full(shape, T::zero(), dtype, device)
    }
    
    /// Create tensor with ones
    pub fn ones(shape: &[usize], dtype: CandleDataType, device: CandleDevice) -> Self {
        Self::full(shape, T::one(), dtype, device)
    }
    
    /// Create tensor with random values
    pub fn randn(shape: &[usize], dtype: CandleDataType, device: CandleDevice) -> Self {
        let size = shape.iter().product();
        let data = Array1::from_vec(
            (0..size).map(|_| T::from(fastrand::f32()).unwrap()).collect()
        );
        
        Self::new(data, false, dtype, device)
    }
    
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get tensor device
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }
    
    /// Get tensor data type
    pub fn dtype(&self) -> CandleDataType {
        self.dtype
    }
    
    /// Check if tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Set gradient requirement
    pub fn set_requires_grad(&mut self, requiresgrad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self._grad = None;
        }
    }
    
    /// Get gradient tensor
    pub fn grad(&self) -> Option<&Array1<T>> {
        self.grad.as_ref()
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Array1::zeros(self.data.len()));
        }
    }
    
    /// Detach tensor from computation graph
    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            grad: None,
            requires_grad: false,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            node_id: None,
            engine: None,
        }
    }
    
    /// Move tensor to device
    pub fn to_device(&mut self, device: CandleDevice) -> Result<()> {
        match (&self.device, &device) {
            (CandleDevice::Cpu, CandleDevice::Cuda(_)) => {
                // CPU to CUDA transfer
                self.device = device;
                Ok(())
            }
            (CandleDevice::Cuda(_), CandleDevice::Cpu) => {
                // CUDA to CPU transfer
                self.device = device;
                Ok(())
            }
            (CandleDevice::Cpu, CandleDevice::Metal(_)) => {
                // CPU to Metal transfer
                self.device = device;
                Ok(())
            }
            _ => {
                // Same device or unsupported transfer
                self.device = device;
                Ok(())
            }
        }
    }
    
    /// Convert tensor data type
    pub fn to_dtype(&mut self, dtype: CandleDataType) -> Result<()> {
        if self.dtype == dtype {
            return Ok(());
        }
        
        // Perform data type conversion
        match (self.dtype, dtype) {
            (CandleDataType::F32, CandleDataType::F16) => {
                // F32 to F16 conversion (simulated)
                self.dtype = dtype;
            }
            (CandleDataType::F16, CandleDataType::F32) => {
                // F16 to F32 conversion (simulated)
                self.dtype = dtype;
            }
            (CandleDataType::F32, CandleDataType::BF16) => {
                // F32 to BF16 conversion (simulated)
                self.dtype = dtype;
            }
            (CandleDataType::BF16, CandleDataType::F32) => {
                // BF16 to F32 conversion (simulated)
                self.dtype = dtype;
            }
            _ => {
                // Other conversions (simulated)
                self.dtype = dtype;
            }
        }
        
        Ok(())
    }
    
    /// Reshape tensor
    pub fn reshape(&mut self, newshape: &[usize]) -> Result<()> {
        let new_size: usize = newshape.iter().product();
        if new_size != self.data.len() {
            return Err(OptimError::DimensionMismatch {
                expected: vec![self.data.len()],
                actual: vec![new_size],
            });
        }
        
        self.shape = newshape.to_vec();
        Ok(())
    }
    
    /// Get tensor data as slice
    pub fn as_slice(&self) -> Result<&[T]> {
        self.data.as_slice().ok_or_else(|| {
            OptimError::InvalidInput("Tensor data is not contiguous".to_string())
        })
    }
    
    /// Get mutable tensor data as slice
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        self.data.as_slice_mut().ok_or_else(|| {
            OptimError::InvalidInput("Tensor data is not contiguous".to_string())
        })
    }
}

impl<T: Float + Default + Clone> CandleOptimizer<T> {
    /// Create new Candle optimizer
    pub fn new(config: CandleOptimizerConfig, device: CandleDevice) -> Result<Self> {
        let autodiff_config = AutodiffConfig {
            enable_forward_mode: true,
            enable_reverse_mode: true,
            enable_hessian: true,
            gradient_checkpointing: config.gradient_checkpointing,
            checkpoint_chunk_size: 1000,
            ..Default::default()
        };
        
        let engine = Arc::new(Mutex::new(AutodiffEngine::new(autodiff_config)));
        
        // Enable candle integration
        if let Ok(mut eng) = engine.lock() {
            eng.integrate_with_candle(true)?;
        }
        
        let mixed_precision = MixedPrecisionConfig {
            enabled: config.auto_mixed_precision,
            forward_precision: if config.auto_mixed_precision { CandleDataType::F16 } else { CandleDataType::F32 },
            backward_precision: CandleDataType::F32,
            param_precision: CandleDataType::F32,
            optimizer_precision: CandleDataType::F32,
            loss_scaling: if config.dynamic_loss_scaling {
                LossScalingStrategy::Dynamic {
                    init_scale: config.loss_scale,
                    growth_factor: 2.0,
                    backoff_factor: 0.5,
                    growth_interval: 2000,
                }
            } else {
                LossScalingStrategy::Fixed(_config.loss_scale)
            },
        };
        
        Ok(Self {
            engine,
            tensors: HashMap::new(),
            config,
            device,
            grad_buffer: HashMap::new(),
            mixed_precision,
        })
    }
    
    /// Register a tensor for optimization
    pub fn register_tensor(&mut self, name: &str, tensor: CandleTensor<T>) -> Result<()> {
        self.tensors.insert(name.to_string(), tensor);
        Ok(())
    }
    
    /// Get registered tensor
    pub fn get_tensor(&self, name: &str) -> Option<&CandleTensor<T>> {
        self.tensors.get(name)
    }
    
    /// Get mutable registered tensor
    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut CandleTensor<T>> {
        self.tensors.get_mut(name)
    }
    
    /// Compute gradients for all registered tensors
    pub fn backward(&mut self, losstensor: &str) -> Result<()> {
        let loss_tensor = self.tensors.get(loss_tensor)
            .ok_or_else(|| OptimError::InvalidInput(format!("Loss _tensor '{}' not found", loss_tensor)))?;
        
        if let Some(node_id) = loss_tensor.node_id {
            if let Ok(mut engine) = self.engine.lock() {
                let gradients = engine.backward(node_id)?;
                
                // Apply loss scaling if using mixed precision
                let scale_factor = self.get_loss_scale();
                
                // Distribute gradients to tensors
                for (name_tensor) in &mut self.tensors {
                    if tensor.requires_grad && tensor.node_id.is_some() {
                        let node_id = tensor.node_id.unwrap();
                        if node_id < gradients.len() {
                            let scaled_grad = gradients[node_id] / scale_factor;
                            
                            if self.config.gradient_accumulation {
                                // Accumulate gradients
                                if let Some(existing_grad) = self.grad_buffer.get_mut(name) {
                                    *existing_grad = &*existing_grad + &Array1::from_elem(1, scaled_grad);
                                } else {
                                    self.grad_buffer.insert(name.clone(), Array1::from_elem(_tensor.data.len(), scaled_grad));
                                }
                            } else {
                                tensor.grad = Some(Array1::from_elem(_tensor.data.len(), scaled_grad));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply gradient clipping
    pub fn clip_gradients(&mut self) -> Result<()> {
        if let Some(clip_value) = self.config.gradient_clip_value {
            for tensor in self.tensors.values_mut() {
                if let Some(ref mut grad) = tensor.grad {
                    grad.mapv_inplace(|g| g.max(-num_traits::cast::cast(clip_value).unwrap_or_else(|| T::zero())).min(num_traits::cast::cast(clip_value).unwrap_or_else(|| T::zero())));
                }
            }
        }
        
        if let Some(clip_norm) = self.config.gradient_clip_norm {
            // Compute total gradient norm
            let mut total_norm_squared = T::zero();
            for tensor in self.tensors.values() {
                if let Some(ref grad) = tensor.grad {
                    total_norm_squared = total_norm_squared + grad.mapv(|g| g * g).sum();
                }
            }
            
            let total_norm = total_norm_squared.sqrt();
            let clip_norm_t = num_traits::cast::cast(clip_norm).unwrap_or_else(|| T::zero());
            
            if total_norm > clip_norm_t {
                let scale_factor = clip_norm_t / total_norm;
                for tensor in self.tensors.values_mut() {
                    if let Some(ref mut grad) = tensor.grad {
                        grad.mapv_inplace(|g| g * scale_factor);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        for tensor in self.tensors.values_mut() {
            tensor.zero_grad();
        }
        self.grad_buffer.clear();
    }
    
    /// Apply gradient accumulation
    pub fn accumulate_gradients(&mut self) -> Result<()> {
        if !self.config.gradient_accumulation {
            return Ok(());
        }
        
        for (name, accumulated_grad) in &self.grad_buffer {
            if let Some(tensor) = self.tensors.get_mut(name) {
                let scale = T::one() / num_traits::cast::cast(self.config.accumulation_steps).unwrap_or_else(|| T::zero());
                tensor.grad = Some(accumulated_grad * scale);
            }
        }
        
        Ok(())
    }
    
    /// Get current loss scale factor
    fn get_loss_scale(&self) -> T {
        match &self.mixed_precision.loss_scaling {
            LossScalingStrategy::Fixed(scale) => num_traits::cast::cast(*scale).unwrap_or_else(|| T::zero()),
            LossScalingStrategy::Dynamic { init_scale, .. } => num_traits::cast::cast(*init_scale).unwrap_or_else(|| T::zero()),
            LossScalingStrategy::Adaptive { .. } => num_traits::cast::cast(1024.0).unwrap_or_else(|| T::zero()), // Default adaptive scale
        }
    }
    
    /// Update loss scaling for dynamic strategies
    pub fn update_loss_scale(&mut self, overflowdetected: bool) -> Result<()> {
        match &mut self.mixed_precision.loss_scaling {
            LossScalingStrategy::Dynamic {
                init_scale,
                growth_factor,
                backoff_factor,
                growth_interval,
            } => {
                if overflow_detected {
                    *init_scale *= *backoff_factor;
                    *init_scale = init_scale.max(1.0); // Minimum scale
                } else {
                    // Grow scale periodically if no overflow
                    *init_scale *= *growth_factor;
                    *init_scale = init_scale.min(65536.0); // Maximum scale
                }
            }
            LossScalingStrategy::Adaptive {
                target_gradient_norm,
                adaptation_rate,
            } => {
                // Compute current gradient norm
                let mut total_norm_squared = 0.0f32;
                for tensor in self.tensors.values() {
                    if let Some(ref grad) = tensor.grad {
                        total_norm_squared += grad.mapv(|g| g.to_f32().unwrap().powi(2)).sum();
                    }
                }
                
                let current_norm = total_norm_squared.sqrt();
                let ratio = *target_gradient_norm / current_norm;
                
                // Update adaptation (simplified)
                *adaptation_rate = (*adaptation_rate * 0.99 + ratio * 0.01).max(0.1).min(10.0);
            }
            _ => {} // Fixed scaling doesn't need updates
        }
        
        Ok(())
    }
    
    /// Enable automatic mixed precision
    pub fn enable_amp(&mut self, enabled: bool) {
        self.mixed_precision.enabled = enabled;
        self.config.auto_mixed_precision = enabled;
    }
    
    /// Check for gradient overflow (for mixed precision)
    pub fn check_gradient_overflow(&self) -> bool {
        for tensor in self.tensors.values() {
            if let Some(ref grad) = tensor.grad {
                for &g in grad.iter() {
                    if !g.is_finite() {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    /// Create tensor operation in computation graph
    pub fn create_operation(
        &mut self,
        op_name: &str,
        inputs: &[&str],
        output_name: &str,
    ) -> Result<()> {
        // Get input tensors and their node IDs
        let mut input_ids = Vec::new();
        for input_name in inputs {
            if let Some(tensor) = self.tensors.get(*input_name) {
                if let Some(node_id) = tensor.node_id {
                    input_ids.push(node_id);
                }
            }
        }
        
        // Create output tensor
        let output_tensor = self.tensors.get(output_name)
            .ok_or_else(|| OptimError::InvalidInput(format!("Output tensor '{}' not found", output_name)))?;
        
        // Register operation in autodiff engine
        if let Ok(mut engine) = self.engine.lock() {
            let output_value = output_tensor.data[0]; // Simplified
            let local_grads = vec![T::one(); input_ids.len()]; // Simplified
            
            let output_id = engine.register_candle_operation(
                op_name,
                &input_ids,
                output_value,
                &local_grads,
            );
            
            // Update output tensor with node ID
            if let Some(tensor) = self.tensors.get_mut(output_name) {
                tensor.node_id = Some(output_id);
            }
        }
        
        Ok(())
    }
    
    /// Optimize computation graph for performance
    pub fn optimize_graph(&mut self) -> Result<()> {
        if let Ok(mut engine) = self.engine.lock() {
            engine.optimize_for_candle()?;
        }
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let mut total_memory = 0;
        let mut gradient_memory = 0;
        
        for tensor in self.tensors.values() {
            total_memory += tensor.data.len() * std::mem::size_of::<T>();
            if let Some(ref grad) = tensor.grad {
                gradient_memory += grad.len() * std::mem::size_of::<T>();
            }
        }
        
        MemoryStats {
            total_tensor_memory: total_memory,
            gradient_memory,
            buffer_memory: self.grad_buffer.len() * std::mem::size_of::<Array1<T>>(),
            peak_memory: total_memory + gradient_memory, // Simplified
        }
    }
    
    /// Save optimizer state
    pub fn save_state(&self) -> Result<CandleOptimizerState<T>> {
        let tensor_states: HashMap<String, TensorState<T>> = self.tensors
            .iter()
            .map(|(name, tensor)| {
                let state = TensorState {
                    data: tensor.data.clone(),
                    grad: tensor.grad.clone(),
                    requires_grad: tensor.requires_grad,
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    device: tensor.device.clone(),
                };
                (name.clone(), state)
            })
            .collect();
        
        Ok(CandleOptimizerState {
            tensor_states,
            config: self.config.clone(),
            mixed_precision: self.mixed_precision.clone(),
            device: self.device.clone(),
        })
    }
    
    /// Load optimizer state
    pub fn load_state(&mut self, state: CandleOptimizerState<T>) -> Result<()> {
        self.tensors.clear();
        
        for (name, tensor_state) in state.tensor_states {
            let tensor = CandleTensor {
                data: tensor_state.data,
                grad: tensor_state.grad,
                requires_grad: tensor_state.requires_grad,
                shape: tensor_state.shape,
                dtype: tensor_state.dtype,
                device: tensor_state.device,
                node_id: None,
                engine: Some(self.engine.clone()),
            };
            
            self.tensors.insert(name, tensor);
        }
        
        self.config = state.config;
        self.mixed_precision = state.mixed_precision;
        self.device = state.device;
        
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory used by tensors (bytes)
    pub total_tensor_memory: usize,
    
    /// Memory used by gradients (bytes)
    pub gradient_memory: usize,
    
    /// Memory used by internal buffers (bytes)
    pub buffer_memory: usize,
    
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
}

/// Saved tensor state
#[derive(Debug, Clone)]
pub struct TensorState<T: Float> {
    pub data: Array1<T>,
    pub grad: Option<Array1<T>>,
    pub requires_grad: bool,
    pub shape: Vec<usize>,
    pub dtype: CandleDataType,
    pub device: CandleDevice,
}

/// Saved optimizer state
#[derive(Debug, Clone)]
pub struct CandleOptimizerState<T: Float> {
    pub tensor_states: HashMap<String, TensorState<T>>,
    pub config: CandleOptimizerConfig,
    pub mixed_precision: MixedPrecisionConfig,
    pub device: CandleDevice,
}

impl Default for CandleOptimizerConfig {
    fn default() -> Self {
        Self {
            auto_mixed_precision: false,
            loss_scale: 1024.0,
            dynamic_loss_scaling: true,
            gradient_clip_value: None,
            gradient_clip_norm: Some(1.0),
            gradient_accumulation: false,
            accumulation_steps: 1,
            gradient_checkpointing: false,
            checkpoint_segments: 1,
            enable_fusion: true,
            memory_optimization: MemoryOptimizationLevel::Basic,
        }
    }
}

impl Default for CandleDevice {
    fn default() -> Self {
        CandleDevice::Cpu
    }
}

/// Utility functions for Candle integration
pub mod utils {
    use super::*;
    
    /// Convert ndarray to Candle tensor
    pub fn from_ndarray<T: Float + Default + Clone>(
        array: Array1<T>,
        dtype: CandleDataType,
        device: CandleDevice,
    ) -> CandleTensor<T> {
        CandleTensor::new(array, false, dtype, device)
    }
    
    /// Convert Candle tensor to ndarray
    pub fn to_ndarray<T: Float + Clone>(tensor: &CandleTensor<T>) -> Array1<T> {
        tensor.data.clone()
    }
    
    /// Create tensor from scalar
    pub fn scalar<T: Float + Default + Clone>(
        value: T,
        dtype: CandleDataType,
        device: CandleDevice,
    ) -> CandleTensor<T> {
        CandleTensor::new(Array1::from_elem(1, value), false, dtype, device)
    }
    
    /// Create tensor from vector
    pub fn from_vec<T: Float + Default + Clone>(
        data: Vec<T>,
        dtype: CandleDataType,
        device: CandleDevice,
    ) -> CandleTensor<T> {
        CandleTensor::new(Array1::from_vec(data), false, dtype, device)
    }
    
    /// Check if two tensors are on the same device
    pub fn same_device<T: Float>(tensor1: &CandleTensor<T>, tensor2: &CandleTensor<T>) -> bool {
        tensor1.device == tensor2.device
    }
    
    /// Get device name as string
    pub fn device_name(device: &CandleDevice) -> String {
        match _device {
            CandleDevice::Cpu => "cpu".to_string(),
            CandleDevice::Cuda(id) => format!("cuda:{}", id),
            CandleDevice::Metal(id) => format!("metal:{}", id),
        }
    }
    
    /// Get data type size in bytes
    pub fn dtype_size(dtype: CandleDataType) -> usize {
        match _dtype {
            CandleDataType::F16 | CandleDataType::BF16 => 2,
            CandleDataType::F32 | CandleDataType::U32 => 4,
            CandleDataType::F64 | CandleDataType::I64 => 8,
            CandleDataType::U8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle_tensor_creation() {
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let tensor = CandleTensor::new(data, true, CandleDataType::F32, CandleDevice::Cpu);
        
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.dtype(), CandleDataType::F32);
        assert_eq!(tensor.device(), &CandleDevice::Cpu);
        assert!(tensor.requires_grad());
    }
    
    #[test]
    fn test_candle_tensor_operations() {
        let mut tensor = CandleTensor::zeros(&[5], CandleDataType::F32, CandleDevice::Cpu);
        
        tensor.set_requires_grad(true);
        assert!(tensor.requires_grad());
        
        tensor.zero_grad();
        assert!(tensor.grad().is_some());
        
        let detached = tensor.detach();
        assert!(!detached.requires_grad());
    }
    
    #[test]
    fn test_candle_optimizer_creation() {
        let config = CandleOptimizerConfig::default();
        let optimizer = CandleOptimizer::<f32>::new(config, CandleDevice::Cpu);
        assert!(optimizer.is_ok());
    }
    
    #[test]
    fn test_mixed_precision_config() {
        let mut config = MixedPrecisionConfig {
            enabled: true,
            forward_precision: CandleDataType::F16,
            backward_precision: CandleDataType::F32,
            param_precision: CandleDataType::F32,
            optimizer_precision: CandleDataType::F32,
            loss_scaling: LossScalingStrategy::Fixed(1024.0),
        };
        
        assert!(config.enabled);
        assert_eq!(config.forward_precision, CandleDataType::F16);
        assert_eq!(config.backward_precision, CandleDataType::F32);
    }
    
    #[test]
    fn test_utils() {
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let tensor = utils::from_ndarray(data.clone(), CandleDataType::F32, CandleDevice::Cpu);
        let recovered = utils::to_ndarray(&tensor);
        
        assert_eq!(data, recovered);
    }
    
    #[test]
    fn test_device_operations() {
        let mut tensor = CandleTensor::zeros(&[3], CandleDataType::F32, CandleDevice::Cpu);
        
        let result = tensor.to_device(CandleDevice::Cuda(0));
        assert!(result.is_ok());
        assert_eq!(tensor.device(), &CandleDevice::Cuda(0));
    }
    
    #[test]
    fn test_dtype_conversion() {
        let mut tensor = CandleTensor::ones(&[2], CandleDataType::F32, CandleDevice::Cpu);
        
        let result = tensor.to_dtype(CandleDataType::F16);
        assert!(result.is_ok());
        assert_eq!(tensor.dtype(), CandleDataType::F16);
    }
}
