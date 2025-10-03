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

//! Example demonstrating the array protocol operations with different array types.

use scirs2_core::array_protocol::{
    self, add, matmul, reshape, sum, transpose, DistributedBackend, DistributedConfig,
    DistributedNdarray, DistributionStrategy, GPUBackend, GPUConfig, GPUNdarray, NdarrayWrapper,
};
use scirs2_core::ndarray_ext::{Array2, Ix2};

#[allow(dead_code)]
fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Array Protocol Operations Example");
    println!("================================");

    // Create regular ndarrays
    let a = Array2::<f64>::eye(3);
    let b = Array2::<f64>::ones((3, 3));

    println!("\nOriginal arrays:");
    println!("A =\n{}", a);
    println!("B =\n{}", b);

    // Wrap in NdarrayWrapper
    let wrapped_a = NdarrayWrapper::new(a.clone());
    let wrapped_b = NdarrayWrapper::new(b.clone());

    // 1. Basic operations with regular arrays
    println!("\n1. Basic operations with regular arrays:");

    // Matrix multiplication
    match matmul(&wrapped_a, &wrapped_b) {
        Ok(result) => {
            if let Some(result_array) = result.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("A * B =\n{}", result_array.as_array());
            } else {
                println!("Matrix multiplication result is not the expected type");
            }
        }
        Err(e) => println!("Error in matrix multiplication: {}", e),
    }

    // Addition
    match add(&wrapped_a, &wrapped_b) {
        Ok(result) => {
            if let Some(result_array) = result.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("A + B =\n{}", result_array.as_array());
            } else {
                println!("Addition result is not the expected type");
            }
        }
        Err(e) => println!("Error in addition: {}", e),
    }

    // Transpose
    match transpose(&wrapped_a) {
        Ok(result) => {
            if let Some(result_array) = result.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("transpose(A) =\n{}", result_array.as_array());
            } else {
                println!("Transpose result is not the expected type");
            }
        }
        Err(e) => println!("Error in transpose: {}", e),
    }

    // Sum
    match sum(&wrapped_a, None) {
        Ok(result) => {
            if let Some(sum_value) = result.downcast_ref::<f64>() {
                println!("sum(A) = {}", sum_value);
            } else {
                println!("Sum result is not a f64 type");
            }
        }
        Err(e) => println!("Error in sum: {}", e),
    }

    // Reshape
    match reshape(&wrapped_a, &[9]) {
        Ok(result) => {
            // When reshaping to 1D, we need to use Ix1
            if let Some(result_array) = result
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix1>>()
            {
                println!("reshape(A, [9]) = {:?}", result_array.as_array());
            } else {
                println!("Reshape result is not the expected type");
            }
        }
        Err(e) => println!("Error in reshape: {}", e),
    }

    // 2. Operations with GPU arrays
    println!("\n2. Operations with GPU arrays:");

    // Create GPU arrays
    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    let gpu_a = GPUNdarray::new(a.clone(), gpu_config.clone());
    let gpu_b = GPUNdarray::new(b.clone(), gpu_config);

    println!(
        "Created GPU arrays with shape {:?} and {:?}",
        gpu_a.shape(),
        gpu_b.shape()
    );

    // Matrix multiplication with GPU arrays
    match matmul(&gpu_a, &gpu_b) {
        Ok(result) => {
            if let Some(gpu_result) = result.as_any().downcast_ref::<GPUNdarray<f64, Ix2>>() {
                println!("GPU matmul result shape: {:?}", gpu_result.shape());
            } else {
                println!("GPU matmul result is not the expected type");
            }
        }
        Err(e) => println!("Error in GPU matrix multiplication: {}", e),
    }

    // Addition with GPU arrays
    match add(&gpu_a, &gpu_b) {
        Ok(result) => {
            if let Some(gpu_result) = result.as_any().downcast_ref::<GPUNdarray<f64, Ix2>>() {
                println!("GPU add result shape: {:?}", gpu_result.shape());
            } else {
                println!("GPU add result is not the expected type");
            }
        }
        Err(e) => println!("Error in GPU addition: {}", e),
    }

    // 3. Operations with distributed arrays
    println!("\n3. Operations with distributed arrays:");

    // Create distributed arrays
    let dist_config = DistributedConfig {
        chunks: 2,
        balance: true,
        strategy: DistributionStrategy::RowWise,
        backend: DistributedBackend::Threaded,
    };

    let dist_a = DistributedNdarray::from_array(&a, dist_config.clone());
    let dist_b = DistributedNdarray::from_array(&b, dist_config);

    println!(
        "Created distributed arrays with {} and {} chunks",
        dist_a.num_chunks(),
        dist_b.num_chunks()
    );

    // Matrix multiplication with distributed arrays
    match matmul(&dist_a, &dist_b) {
        Ok(result) => {
            if let Some(dist_result) = result
                .as_any()
                .downcast_ref::<DistributedNdarray<f64, Ix2>>()
            {
                println!("Distributed matmul result shape: {:?}", dist_result.shape());
            } else {
                println!("Distributed matmul result is not the expected type");
            }
        }
        Err(e) => println!("Error in distributed matrix multiplication: {}", e),
    }

    // Addition with distributed arrays
    match add(&dist_a, &dist_b) {
        Ok(result) => {
            if let Some(dist_result) = result
                .as_any()
                .downcast_ref::<DistributedNdarray<f64, Ix2>>()
            {
                println!("Distributed add result shape: {:?}", dist_result.shape());
            } else {
                println!("Distributed add result is not the expected type");
            }
        }
        Err(e) => println!("Error in distributed addition: {}", e),
    }

    // 4. Mixed array type operations
    println!("\n4. Mixed array type operations:");

    // GPU array + distributed array
    match add(&gpu_a, &dist_b) {
        Ok(_) => println!("Mixed add (GPU + Distributed) completed successfully"),
        Err(e) => println!("Error in mixed add (GPU + Distributed): {}", e),
    }

    // Regular array + GPU array
    match matmul(&wrapped_a, &gpu_b) {
        Ok(_) => println!("Mixed matmul (Regular + GPU) completed successfully"),
        Err(e) => println!("Error in mixed matmul (Regular + GPU): {}", e),
    }

    println!("\nAll operations completed successfully!");
}
