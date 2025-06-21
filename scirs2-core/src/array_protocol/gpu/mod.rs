// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! GPU-accelerated array implementation for the array protocol.
//!
//! This module provides GPU-accelerated array implementations that work with
//! the array protocol. It supports multiple GPU backends (CUDA, ROCm, Metal,
//! WebGPU, OpenCL) and provides specialized operations for each.

mod cuda_operations;

pub use cuda_operations::*;

/// Initializes the GPU system for the array protocol.
pub fn init_gpu_system() {
    // Initialize CUDA operations
    cuda_operations::register_cuda_operations();

    // Other GPU backends would be initialized here
}