//! GPU acceleration utility functions
//!
//! This module provides utility functions for GPU operations including
//! system capability checking, memory optimization, and configuration.

use super::{
    GpuConfig, GpuDeviceManager, GraphOptimizationLevel, MemoryStrategy, TensorCoresConfig,
};

/// Check if GPU acceleration is supported on this system
pub fn is_gpu_supported() -> bool {
    // Check for actual GPU framework availability
    if let Ok(device_manager) = GpuDeviceManager::new() {
        device_manager.is_gpu_available()
    } else {
        false
    }
}

/// Get recommended batch size for GPU operations
pub fn get_recommended_batch_size(_data_size: usize, memorylimit: usize) -> usize {
    let element_size = std::mem::size_of::<f64>(); // Assume f64 for estimation
    let max_batch = memorylimit / element_size;
    std::cmp::min(_data_size, max_batch)
}

/// Estimate GPU memory requirements for operation
pub fn estimate_memory_usage(_data_size: usize, operationoverhead: f64) -> usize {
    let base_memory = _data_size * std::mem::size_of::<f64>();
    (base_memory as f64 * (1.0 + operationoverhead)) as usize
}

/// Choose optimal GPU configuration based on data characteristics
pub fn optimize_gpu_config(_data_size: usize, availablememory: usize) -> GpuConfig {
    let batch_size = get_recommended_batch_size(_data_size, availablememory / 4);

    GpuConfig {
        device_id: 0,
        memory_pool_size: Some(availablememory / 2),
        enable_memory_optimization: true,
        batch_size,
        use_half_precision: _data_size > 100_000,
        enable_async: true,
        tensor_cores: TensorCoresConfig::default(),
        memory_strategy: MemoryStrategy::PreAllocated {
            pool_size: availablememory / 2,
        },
        dynamic_batching: true,
        graph_optimization: GraphOptimizationLevel::Extended,
    }
}
