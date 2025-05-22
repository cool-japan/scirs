// Copyright (c) 2025, SciRS2 Team
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

//! Example demonstrating the array protocol with GPU acceleration.
//!
//! This example shows how to use the array protocol with different GPU backends
//! and how to perform GPU-accelerated operations on arrays.

use ndarray::{Array2, Ix2};
use scirs2_core::array_protocol::{
    self, add,
    auto_device::{AutoDevice, DeviceSelection},
    matmul, multiply, subtract, sum, transpose, GPUBackend, GPUConfig, GPUNdarray, NdarrayWrapper,
};

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Array Protocol GPU Example");
    println!("==========================");

    // Create regular ndarrays for our computation
    let a = Array2::<f64>::eye(3);
    let b = Array2::<f64>::ones((3, 3));
    let c = Array2::<f64>::from_elem((3, 3), 2.0);

    println!("\nOriginal arrays:");
    println!("A =\n{}", a);
    println!("B =\n{}", b);
    println!("C =\n{}", c);

    // Wrap arrays in NdarrayWrapper for CPU operations
    let wrapped_a = NdarrayWrapper::new(a.clone());
    let wrapped_b = NdarrayWrapper::new(b.clone());
    let wrapped_c = NdarrayWrapper::new(c.clone());

    // 1. CPU operations for comparison
    println!("\n1. Operations on CPU:");

    // Matrix multiplication: A * B
    match matmul(&wrapped_a, &wrapped_b) {
        Ok(result) => {
            if let Some(result_array) = result.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("CPU: A * B =\n{}", result_array.as_array());
            }
        }
        Err(e) => println!("Error in CPU matrix multiplication: {}", e),
    }

    // Matrix addition: A + C
    match add(&wrapped_a, &wrapped_c) {
        Ok(result) => {
            if let Some(result_array) = result.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("CPU: A + C =\n{}", result_array.as_array());
            }
        }
        Err(e) => println!("Error in CPU addition: {}", e),
    }

    // 2. Operations with CUDA GPU backend
    println!("\n2. Operations with CUDA GPU backend:");

    // Create GPU configuration for CUDA
    let cuda_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.8,
    };

    // Move arrays to GPU
    let gpu_a = GPUNdarray::new(a.clone(), cuda_config.clone());
    let gpu_b = GPUNdarray::new(b.clone(), cuda_config.clone());
    let gpu_c = GPUNdarray::new(c.clone(), cuda_config);

    println!(
        "Created GPU arrays with shapes {:?}, {:?}, and {:?}",
        gpu_a.shape(),
        gpu_b.shape(),
        gpu_c.shape()
    );

    // Matrix multiplication on GPU: A * B
    match matmul(&gpu_a, &gpu_b) {
        Ok(result) => {
            if let Some(gpu_result) = result.as_any().downcast_ref::<GPUNdarray<f64, Ix2>>() {
                println!("CUDA GPU: A * B shape: {:?}", gpu_result.shape());

                // Convert result back to CPU for display
                match gpu_result.to_cpu() {
                    Ok(cpu_result) => {
                        if let Some(ndarray_result) = cpu_result
                            .as_any()
                            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
                        {
                            println!("CUDA GPU: A * B =\n{}", ndarray_result.as_array());
                        }
                    }
                    Err(e) => println!("Error converting GPU result to CPU: {}", e),
                }
            }
        }
        Err(e) => println!("Error in CUDA GPU matrix multiplication: {}", e),
    }

    // Matrix addition on GPU: A + C
    match add(&gpu_a, &gpu_c) {
        Ok(result) => {
            if let Some(gpu_result) = result.as_any().downcast_ref::<GPUNdarray<f64, Ix2>>() {
                println!("CUDA GPU: A + C shape: {:?}", gpu_result.shape());

                // Convert result back to CPU for display
                match gpu_result.to_cpu() {
                    Ok(cpu_result) => {
                        if let Some(ndarray_result) = cpu_result
                            .as_any()
                            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
                        {
                            println!("CUDA GPU: A + C =\n{}", ndarray_result.as_array());
                        }
                    }
                    Err(e) => println!("Error converting GPU result to CPU: {}", e),
                }
            }
        }
        Err(e) => println!("Error in CUDA GPU addition: {}", e),
    }

    // 3. Using AutoDevice for automatic device selection
    println!("\n3. Using AutoDevice for automatic device selection:");

    // Create an AutoDevice instance
    let auto_device = AutoDevice::new()
        .with_device_selection(DeviceSelection::Automatic)
        .with_gpu_memory_threshold(1024 * 1024) // 1MB threshold
        .build();

    // Wrap arrays using AutoDevice
    let auto_a = auto_device.wrap_array(a.clone());
    let auto_b = auto_device.wrap_array(b.clone());
    let auto_c = auto_device.wrap_array(c.clone());

    println!("Created auto-device arrays");

    // Matrix multiplication using auto-device selection
    match matmul(&auto_a, &auto_b) {
        Ok(result) => {
            println!("AutoDevice: A * B completed successfully");
            // Display where computation was performed
            if auto_device.last_operation_on_gpu() {
                println!("  Operation performed on GPU");
            } else {
                println!("  Operation performed on CPU");
            }
        }
        Err(e) => println!("Error in AutoDevice matrix multiplication: {}", e),
    }

    // 4. Comparing performance
    println!("\n4. Performance comparison:");

    // Create larger arrays for performance testing
    let size = 500;
    let large_a = Array2::<f64>::eye(size);
    let large_b = Array2::<f64>::ones((size, size));

    // Wrap in CPU arrays
    let cpu_large_a = NdarrayWrapper::new(large_a.clone());
    let cpu_large_b = NdarrayWrapper::new(large_b.clone());

    // Wrap in GPU arrays
    let gpu_large_a = GPUNdarray::new(large_a.clone(), cuda_config.clone());
    let gpu_large_b = GPUNdarray::new(large_b.clone(), cuda_config);

    // CPU timing
    println!(
        "Starting CPU matrix multiplication for {}x{} matrices...",
        size, size
    );
    let cpu_start = std::time::Instant::now();
    let _ = matmul(&cpu_large_a, &cpu_large_b);
    let cpu_duration = cpu_start.elapsed();
    println!("CPU computation time: {:?}", cpu_duration);

    // GPU timing
    println!(
        "Starting GPU matrix multiplication for {}x{} matrices...",
        size, size
    );
    let gpu_start = std::time::Instant::now();
    let _ = matmul(&gpu_large_a, &gpu_large_b);
    let gpu_duration = gpu_start.elapsed();
    println!("GPU computation time: {:?}", gpu_duration);

    println!(
        "\nGPU speedup: {:.2}x",
        cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64()
    );

    println!("\nArray Protocol GPU examples completed successfully!");
}
